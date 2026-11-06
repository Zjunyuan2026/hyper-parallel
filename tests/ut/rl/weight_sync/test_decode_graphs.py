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
"""CPU contracts for pinned Ascend decode graph publication and failure propagation."""

from contextlib import ExitStack
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rl.weight_sync import vllm_worker


class TestDecodeGraphs(unittest.TestCase):
    """Exercise the real refresh implementation across mocked NPU capture boundaries."""

    def setUp(self) -> None:
        """Keep graph state local and fail if refresh keeps stale attention-task references."""
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.config = object()
        self.old_graph = Mock()
        self.wrapper = SimpleNamespace(vllm_config=self.config, concrete_aclgraph_entries={
            1: SimpleNamespace(aclgraph=self.old_graph), 4: SimpleNamespace(aclgraph=Mock()),
        })
        self.original_pool = object()
        self.wrapper.graph_pool = self.original_pool
        self.other = SimpleNamespace(vllm_config=object(), concrete_aclgraph_entries={9: object()})
        self.params = SimpleNamespace(**{
            name: {1: [object()], 4: [object()]} for name in ("events", "handles", "attn_params", "workspaces")
        })
        self.monitor = Mock()
        self.acl_graph = SimpleNamespace(_acl_graph_wrappers=[self.wrapper, self.other],
                                         get_graph_params=lambda: self.params)
        modules = {
            "vllm.config": SimpleNamespace(CUDAGraphMode=SimpleNamespace(FULL_DECODE_ONLY="decode")),
            "vllm.compilation.monitor": SimpleNamespace(set_cudagraph_capturing_enabled=self.monitor),
            "vllm_ascend.compilation": SimpleNamespace(acl_graph=self.acl_graph),
        }
        contexts.enter_context(patch.dict(sys.modules, modules))
        contexts.enter_context(patch.object(vllm_worker, "package_version", side_effect=lambda name: {
            "vllm": "0.23.0+empty", "vllm-ascend": "0.23.0.post1",
        }[name]))
        self.device = Mock()
        contexts.enter_context(patch.object(vllm_worker.torch, "npu", self.device, create=True))
        self.runner = SimpleNamespace(vllm_config=self.config,
                                      compilation_config=SimpleNamespace(cudagraph_mode="decode"),
                                      capture_model=self._capture)
        self.worker = SimpleNamespace(model_runner=self.runner, _rl_graph_refresh_pending=True,
                                      _rl_loaded_policy_version=3)

    def _capture(self) -> None:
        """Capture with fresh task metadata while retired graphs keep the shared pool alive."""
        self.assertFalse(self.wrapper.concrete_aclgraph_entries)
        self.old_graph.reset.assert_not_called()
        for name in ("events", "handles", "attn_params"):
            self.assertEqual(getattr(self.params, name), {1: [], 4: []})
        self.assertEqual(self.params.workspaces, {1: None, 4: None})
        self.wrapper.concrete_aclgraph_entries.update({
            size: SimpleNamespace(aclgraph=Mock()) for size in (1, 4)
        })

    def test_refresh_replaces_every_capture_and_task_handle_without_touching_other_models(self) -> None:
        """The new policy becomes graph-ready only after complete recapture and device synchronization."""
        vllm_worker._refresh_ascend_decode_graphs(self.worker)  # pylint: disable=protected-access
        self.old_graph.reset.assert_called_once_with()
        self.assertFalse(self.worker._rl_graph_refresh_pending)  # pylint: disable=protected-access
        self.assertEqual(set(self.other.concrete_aclgraph_entries), {9})
        self.assertEqual(self.device.synchronize.call_count, 2)
        self.device.graph_pool_handle.assert_not_called()
        self.assertIs(self.wrapper.graph_pool, self.original_pool)
        self.monitor.assert_called_once_with(False)

    def test_capture_failure_keeps_policy_unready_and_disables_unexpected_lazy_capture(self) -> None:
        """Never acknowledge graph readiness or silently fall back after a failed capture."""
        self.runner.capture_model = Mock(side_effect=RuntimeError("capture failed"))
        with self.assertRaisesRegex(RuntimeError, "capture failed"):
            vllm_worker._refresh_ascend_decode_graphs(self.worker)  # pylint: disable=protected-access
        self.assertTrue(self.worker._rl_graph_refresh_pending)  # pylint: disable=protected-access
        self.monitor.assert_called_once_with(False)

    def test_incomplete_recapture_and_unreviewed_runtime_are_rejected(self) -> None:
        """A successful capture call alone does not prove every batch descriptor was rebuilt."""
        self.runner.capture_model = Mock()
        with self.assertRaisesRegex(RuntimeError, "every original capture descriptor"):
            vllm_worker._refresh_ascend_decode_graphs(self.worker)  # pylint: disable=protected-access
        self.assertTrue(self.worker._rl_graph_refresh_pending)  # pylint: disable=protected-access
        with patch.object(vllm_worker, "package_version", return_value="0.22.1"):
            with self.assertRaisesRegex(RuntimeError, "pinned"):
                vllm_worker._refresh_ascend_decode_graphs(self.worker)  # pylint: disable=protected-access

    def test_eager_worker_does_not_load_graph_dependencies(self) -> None:
        """The established eager recipe remains independent of graph refresh."""
        self.worker._rl_graph_refresh_pending = False  # pylint: disable=protected-access
        with patch.object(vllm_worker, "package_version", side_effect=AssertionError("unexpected graph dependency")):
            vllm_worker._refresh_ascend_decode_graphs(self.worker)  # pylint: disable=protected-access
        self.device.synchronize.assert_not_called()
