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
"""Verify device-unique HCCL routes and complete bucket acknowledgement coverage."""
from types import SimpleNamespace

import pytest
import torch

import rl.weight_sync.colocated_hccl as module
import rl.weight_sync.vllm_worker as worker_module
from rl.weight_sync.colocated_hccl import ColocatedHCCLWeightTransport, colocated_routes
from rl.weight_sync.ipc import IPCContext, PhysicalRolloutWorker


@pytest.mark.parametrize("count", [2, 4, 8, 16])
def test_routes_are_device_unique_and_cover_every_receiver(count):
    devices = tuple(f"npu-{index}" for index in range(count))
    covered = []
    for owner, receivers in colocated_routes(devices):
        assert devices[owner] not in receivers, f"owner={owner}, receivers={receivers}"
        assert len(set(receivers)) == len(receivers), f"receivers={receivers}"
        covered.extend(receivers)
    assert sorted(covered) == sorted(devices), f"covered={covered}, expected={devices}"


@pytest.mark.parametrize("devices", [(), ("npu-0",), ("npu-0", "npu-0")])
def test_invalid_device_topologies_rejected(devices):
    with pytest.raises(ValueError):
        colocated_routes(devices)


@pytest.mark.parametrize("rank", [0, 1, 15])
def test_prepare_send_two_buckets_and_check_all_workers(monkeypatch, rank):
    devices = tuple(f"npu-{index}" for index in range(16))
    workers = tuple(PhysicalRolloutWorker(i // 4, i % 4, devices[i]) for i in range(16))
    context = IPCContext("http://shared", devices[rank], workers[rank], workers)
    transport = ColocatedHCCLWeightTransport(data_parallel_size=4, tensor_parallel_size=4)
    monkeypatch.setattr(transport._ownership, "prepare_packed", lambda client: context)
    monkeypatch.setattr(transport, "_resolve_endpoint", lambda client: "http://shared")
    transport._endpoint = "http://shared"
    monkeypatch.setattr(module, "coordinator_call", lambda operation, callback: callback())
    monkeypatch.setattr(module, "synchronized_call", lambda operation, callback: callback())
    monkeypatch.setattr(module, "dist", SimpleNamespace(
        get_rank=lambda: rank, get_world_size=lambda: 16,
        all_gather_object=lambda values, unused: values.__setitem__(slice(None), devices),
        broadcast=lambda tensor, src: tensor.fill_(7),
    ))
    handle = SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None),
                             synchronize=lambda: None, current_device=lambda: rank)
    monkeypatch.setattr(torch, "get_device_module", lambda: handle)
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: torch.device("cpu"))
    groups = []
    def initialize(info):
        group = SimpleNamespace(broadcast=lambda tensor, src: groups.append(tensor.clone()))
        return group
    monkeypatch.setattr(transport, "_trainer_init", initialize)
    calls = []
    def rpc(client, endpoint, method, kwargs, action, timeout):
        local = action()
        receivers = kwargs["receiver_devices"]
        calls.append((method, receivers))
        replies = []
        for worker in workers:
            physical = worker.physical_device_id
            active = physical in receivers
            reply = {"dp_rank": worker.dp_rank, "tp_rank": worker.tp_rank, "physical_device_id": physical}
            if method == "init_colocated_packed_weight_group":
                reply.update(joined=active, group_rank=1 + receivers.index(physical) if active else -1)
            else:
                reply.update(received=True, bytes=kwargs["total_bytes"] if active else 0)
            replies.append(reply)
        return local, replies
    monkeypatch.setattr(module, "_rpc_with_collective", rpc)
    endpoint = transport.prepare_packed(None)
    for index in range(2):
        ack = transport.send_packed_bucket(None, endpoint, index, [], 16,
                                           torch.zeros(16, dtype=torch.uint8) if rank == 0 else None, 1)
        assert ack.worker_count == 16, f"ack={ack}"
    assert len(calls) == 6, f"calls={calls}"
    assert len(groups) == (0 if rank == 15 else 2), f"groups={len(groups)}, rank={rank}"
    with pytest.raises(RuntimeError, match="cover every worker"):
        transport._validate_replies([], devices, "received", 16)
    transport.close()
    assert not transport._routes, f"routes={transport._routes}"


@pytest.mark.parametrize("index", [0, 1, 15])
def test_worker_joins_only_its_device_unique_receiver_route(monkeypatch, index):
    """Skipped workers acknowledge without entering another worker's collective."""
    devices = [f"npu-{i}" for i in range(16)]
    monkeypatch.setattr(worker_module, "_rollout_worker_topology", lambda worker: {
        "dp_rank": index // 4, "tp_rank": index % 4, "physical_device_id": devices[index],
    })
    joined = []
    monkeypatch.setattr(worker_module, "_join_hccl_group", lambda *args, **kwargs: joined.append(kwargs))
    result = worker_module.init_colocated_packed_weight_group(
        object(), group_id="route", master_address="127.0.0.1", master_port=12345,
        world_size=16, receiver_devices=devices[1:], producer_device=devices[0],
    )
    assert result["joined"] == (index != 0), f"result={result}"
    assert len(joined) == int(index != 0), f"joined={joined}"
    if joined:
        assert joined[0]["receiver_rank"] == index, f"joined={joined}"
    with pytest.raises(ValueError, match="unique device"):
        worker_module.init_colocated_packed_weight_group(
            object(), group_id="bad", master_address="127.0.0.1", master_port=12345,
            world_size=17, receiver_devices=devices, producer_device=devices[0],
        )
