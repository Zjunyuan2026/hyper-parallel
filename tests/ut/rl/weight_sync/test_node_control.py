# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Node-local HTTP/IPC ownership must retain global failure and weight propagation."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import yaml

from rl.algorithm import build_algorithm
from rl.config.limits import resolve_vllm_automatic_limits
from rl.config.resolve import build_runtime_config, resolve_router_replay_config, resolve_unified_execution_config
from rl.trainer.build import _validate_shared_vllm_metadata
from rl.config.validate import _validate_colocated_vllm, validate_config
from rl.rollout.vllm import VLLMGenerationEngine
from rl.weight_sync import sync
from rl.weight_sync.ipc import IPCContext, IPCWeightTransport, PhysicalRolloutWorker
from rl.weight_sync.vllm_client import VLLMWeightSyncClientMixin, shared_endpoint
from hyper_parallel.rl.examples.consistency.scripts.configure_run import configure


class NodeControlTests(unittest.TestCase):
    """Separate each node's physical resources without separating policy versions."""

    def test_generated_four_node_recipe_passes_production_configuration(self) -> None:
        """Resolve and validate global training/local inference together without building a model."""
        source = (Path(__file__).resolve().parents[4]
                  / "hyper_parallel/rl/examples/consistency/configs/qwen3_moe_4l_cp_rl.yaml")
        environment = {"RUN_MODEL_PATH": "/models/Qwen3-30B-A3B", "TRAIN_TP": "4", "TRAIN_CP": "2",
                       "TRAIN_EP": "4", "ROLLOUT_TP": "2", "ROLLOUT_PCP": "1", "ROLLOUT_EP": "8",
                       "ROLLOUT_LOCAL_WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "16", "CPU_OFFLOAD": "false",
                       "GLOBAL_PROMPT_BATCH_SIZE": "136", "CANDIDATES": "4", "MICRO_BATCH_SIZE": "1",
                       "RESPONSE_MINI_BATCH_SIZE": "68", "FULL_DATASET": "true", "STEPS": "5"}
        with tempfile.TemporaryDirectory() as temporary, patch.dict("os.environ", environment, clear=True), \
             patch.object(torch.accelerator, "current_accelerator", return_value=SimpleNamespace(type="npu")):
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({
                "model_type": "qwen3_moe", "architectures": ["Qwen3MoeForCausalLM"],
                "num_experts": 128, "num_hidden_layers": 48, "hidden_size": 2048,
                "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
            }))
            parquet = root / "train.parquet"
            parquet.touch()
            config = configure(yaml.safe_load(source.read_text()), 64)
            config["model"].update(weights_path=str(root), tokenizer_path=str(root))
            config["data"]["train_path"] = str(parquet)
            config = resolve_vllm_automatic_limits(
                resolve_router_replay_config(resolve_unified_execution_config(config)),
            )
            shared_pik = [entry["replace"]["implementation"]["num_leaves"]
                          for entry in config["train"]["plan_overrides"]
                          if entry.get("replace", {}).get("implementation", {}).get("_target_", "").endswith(
                              "replace_attention_with_pik")]
            self.assertEqual(shared_pik, [4])
            self.assertEqual(config["rollout"]["vllm"]["node_local_world_size"], 16)
            validate_config(config, build_algorithm(config["algorithm"]))
            runtime = build_runtime_config(config)
            self.assertEqual(runtime.fsdp_config.dp_shard_size, 8)
            self.assertEqual(runtime.fsdp_config.edp_shard_size, 16)
            self.assertEqual(config["train"]["prompt_batch_size"], 17)
            self.assertEqual(config["train"]["response_mini_batch_size"], 68)
            config["async_training"] = {}
            with self.assertRaisesRegex(ValueError, "synchronous"):
                validate_config(config, build_algorithm(config["algorithm"]))

    def test_all_ranks_create_node_groups_in_identical_order(self) -> None:
        """Global rank 32 owns the third contiguous group, not the global coordinator."""
        with patch.object(sync.dist, "get_world_size", return_value=64), \
             patch.object(sync.dist, "get_rank", return_value=32), \
             patch.object(sync.dist, "new_group", side_effect=tuple) as create:
            group = sync.create_node_control_group(16)
        self.assertEqual(group, tuple(range(32, 48)))
        self.assertEqual([call.args[0] for call in create.call_args_list],
                         [list(range(start, start + 16)) for start in (0, 16, 32, 48)])

    def test_second_node_owner_controls_its_server(self) -> None:
        """A successful node operation broadcasts locally after checking failures globally."""
        group = object()
        callback = Mock(return_value={"version": 4})
        with patch.object(sync.dist, "get_world_size", return_value=64), \
             patch.object(sync.dist, "get_rank", return_value=0), \
             patch.object(sync.dist, "get_global_rank", return_value=16), \
             patch.object(sync.dist, "all_gather_object") as gather, \
             patch.object(sync.dist, "broadcast_object_list") as broadcast:
            self.assertEqual(sync.coordinator_call("wake", callback, group=group), {"version": 4})
        callback.assert_called_once_with()
        self.assertEqual(len(gather.call_args.args[0]), 64)
        self.assertEqual(gather.call_args.kwargs, {})
        broadcast.assert_called_once_with([{"version": 4}], src=16, group=group)

    def test_peer_node_failure_rejects_every_node(self) -> None:
        """A different node's HTTP error must reject this node before its result is admitted."""
        def gather(records, _error):
            records[32] = "remote node failed to load weights"

        with patch.object(sync.dist, "get_world_size", return_value=64), \
             patch.object(sync.dist, "get_rank", return_value=1), \
             patch.object(sync.dist, "all_gather_object", side_effect=gather), \
             patch.object(sync.dist, "broadcast_object_list") as broadcast:
            with self.assertRaisesRegex(RuntimeError, "remote node failed"):
                sync.coordinator_call("publish", Mock(), group=object())
        broadcast.assert_not_called()

    def test_stale_policy_is_detected_inside_global_error_propagation(self) -> None:
        """A stale node must fail before the coordinator broadcasts its version locally."""
        client = Mock(spec=VLLMWeightSyncClientMixin)
        lifecycle = sync.ActorRolloutWeightSync("model", "colocated", lambda: client, None, control_group=object())

        def gather(records, error):
            records[16] = error

        with patch.object(sync.dist, "get_world_size", return_value=64), \
             patch.object(sync.dist, "get_rank", return_value=0), \
             patch.object(sync.dist, "all_gather_object", side_effect=gather), \
             patch.object(sync.dist, "broadcast_object_list") as broadcast, \
             patch.object(sync, "committed_policy_version", return_value=7):
            with self.assertRaisesRegex(RuntimeError, "failed on at least one rank.*published version"):
                lifecycle.generation_version(client)
        broadcast.assert_not_called()

    def test_admission_uses_local_request_owners(self) -> None:
        """The sixth global DP owner is the second owner on its local node."""
        engine = object.__new__(VLLMGenerationEngine)
        engine._config = {"node_local_world_size": 16}
        engine.configure_trainer_tensor_parallel(group=object(), tp_rank=0, tp_size=8,
                                                request_rank=5, request_size=8)
        self.assertEqual((engine._trainer_request_rank, engine._trainer_request_size), (1, 2))

    def test_metadata_allows_per_node_endpoints_but_rejects_partial_node(self) -> None:
        """Each local group agrees on its own complete physical device mapping."""
        devices = ",".join(map(str, range(16)))
        records = [(16, "127.0.0.1", 8100 + node, devices) for node in range(4) for _ in range(16)]
        _validate_shared_vllm_metadata(records, "colocated", 64, 64, node_local_world_size=16)
        records[17] = (8, "127.0.0.1", 8101, devices)
        with self.assertRaisesRegex(ValueError, "local_world_size"):
            _validate_shared_vllm_metadata(records, "colocated", 64, 64, node_local_world_size=16)

    def test_config_keeps_train_request_groups_on_one_node(self) -> None:
        """Global DP may span nodes; each TP x CP request group must be local."""
        vllm = {"node_local_world_size": 16, "tensor_parallel_size": 2}
        accelerator = {"tp": 4, "cp": 2}
        with patch.dict("os.environ", {"LOCAL_WORLD_SIZE": "16"}):
            _validate_colocated_vllm(vllm, accelerator, 64, 8, 2)
            with self.assertRaisesRegex(ValueError, "TP x CP"):
                _validate_colocated_vllm(vllm, {"tp": 16, "cp": 2}, 64, 8, 2)

    def test_endpoint_and_ipc_handles_are_only_gathered_within_node(self) -> None:
        """Repeated device indices on other hosts must never enter this IPC payload."""
        group = object()
        transport = IPCWeightTransport(data_parallel_size=2, control_group=group)
        workers = tuple(PhysicalRolloutWorker(index, 0, f"node-device-{index}") for index in range(2))
        context = IPCContext("http://localhost:8100", workers[0].physical_device_id, workers[0], workers)

        def gather(records, value, *, group):  # pylint: disable=redefined-outer-name
            self.assertIs(group, transport._control_group)
            records[:] = ([value, value] if isinstance(value, str)
                          else [{workers[0].physical_device_id: "local-handle"},
                                {workers[1].physical_device_id: "other-local-handle"}])

        with patch.object(sync.dist, "get_world_size", return_value=2), \
             patch.object(sync.dist, "all_gather_object", side_effect=gather), \
             patch("rl.weight_sync.ipc.synchronized_call", side_effect=lambda _name, callback: callback()), \
             patch("rl.weight_sync.ipc.tensor_ipc_rebuild_args", return_value="local-handle"):
            self.assertEqual(shared_endpoint(SimpleNamespace(base_url=context.endpoint), group=group), context.endpoint)
            handles = transport._collect_handles(context, object(), {worker.physical_device_id for worker in workers})
        self.assertEqual(set(handles), {"node-device-0", "node-device-1"})

    def test_packed_weight_tensor_still_broadcasts_from_global_rank_zero(self) -> None:
        """Node-local IPC must not turn the full-policy tensor broadcast into a local operation."""
        transport = IPCWeightTransport(control_group=object())
        worker = PhysicalRolloutWorker(0, 0, "local-device")
        context = IPCContext("http://localhost", worker.physical_device_id, worker, (worker,))
        device = SimpleNamespace(current_device=lambda: 0,
                                 current_stream=lambda: SimpleNamespace(synchronize=lambda: None))
        with patch.object(sync.dist, "get_rank", return_value=16), \
             patch.object(sync.dist, "broadcast") as broadcast, \
             patch.object(torch.accelerator, "current_accelerator", return_value=torch.device("cpu")), \
             patch.object(torch, "get_device_module", return_value=device), \
             patch("rl.weight_sync.ipc.synchronized_call", side_effect=lambda _name, callback: callback()), \
             patch.object(transport, "_collect_handles", return_value={}), patch.object(transport, "_receive"):
            transport.send_packed_bucket(Mock(), context, 0, [], 4, None, 1)
        self.assertEqual(broadcast.call_args.kwargs, {"src": 0})
        self.assertEqual(broadcast.call_args.args[0].numel(), 4)
