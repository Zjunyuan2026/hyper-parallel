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
"""Colocated HCCL publication with unique devices inside each communicator."""
import logging
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist

from rl.weight_sync.hccl import HCCLWeightTransport, _open_port, _rpc_with_collective
from rl.weight_sync.ipc import IPCWeightTransport
from rl.weight_sync.packed_weight import PackedWeightAck
from rl.weight_sync.sync import coordinator_call, synchronized_call


_LOGGER = logging.getLogger(__name__)


def colocated_routes(devices: tuple) -> tuple:
    """Partition receivers between producers on two distinct devices."""
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("Colocated HCCL requires at least two distinct devices")
    return ((0, devices[1:]), (1, devices[:1]))


class ColocatedHCCLWeightTransport(HCCLWeightTransport):
    """Publish through two groups without placing two ranks on the same device."""

    def __init__(self, *, data_parallel_size: int = 1, tensor_parallel_size: int = 1) -> None:
        super().__init__(data_parallel_size=data_parallel_size, tensor_parallel_size=tensor_parallel_size)
        # Reuse physical ownership discovery only; no IPC handle or shared storage is exported.
        self._ownership = IPCWeightTransport(
            data_parallel_size=data_parallel_size, tensor_parallel_size=tensor_parallel_size,
        )
        self._physical_context = None
        self._routes = []
        self._route_groups = {}

    def _validate_replies(self, replies: Any, receivers: tuple, key: str, total_bytes: int = 0) -> None:
        """Require one exact physical acknowledgement from every worker, including skips."""
        expected = {(worker.dp_rank, worker.tp_rank): worker.physical_device_id
                    for worker in self._physical_context.workers}
        observed = set()
        if not isinstance(replies, list):
            raise RuntimeError("Colocated HCCL returned invalid replies")
        for reply in replies:
            coordinate = (int(reply["dp_rank"]), int(reply["tp_rank"]))
            physical = reply.get("physical_device_id")
            if coordinate in observed or expected.get(coordinate) != physical:
                raise RuntimeError(f"Colocated HCCL worker identity mismatch: {reply}")
            observed.add(coordinate)
            active = physical in receivers
            if key == "joined":
                rank = 1 + receivers.index(physical) if active else -1
                if reply.get("joined") is not active or reply.get("group_rank") != rank:
                    raise RuntimeError(f"Colocated HCCL group membership mismatch: {reply}")
            elif reply.get("received") is not True or reply.get("bytes") != (total_bytes if active else 0):
                raise RuntimeError(f"Colocated HCCL bucket acknowledgement mismatch: {reply}")
        if observed != set(expected):
            raise RuntimeError("Colocated HCCL replies do not cover every worker")

    def prepare_packed(self, client: Any) -> str:
        """Initialize a main group and a second route for the first producer's device."""
        endpoint = self._resolve_endpoint(client)
        if self._routes:
            return endpoint
        context = self._ownership.prepare_packed(client)
        self._physical_context = context
        devices = tuple(worker.physical_device_id for worker in context.workers)
        actual = [None] * dist.get_world_size()
        dist.all_gather_object(actual, context.physical_device_id)
        if tuple(actual) != devices:
            raise ValueError("Colocated HCCL requires identical Trainer and rollout physical device order")
        for owner, receivers in colocated_routes(devices):
            port = coordinator_call("colocated HCCL rendezvous", _open_port)
            group_id = f"hyper-colocated-p{port}-owner{owner}"
            init = {"master_address": "127.0.0.1", "master_port": port, "world_size": 1 + len(receivers)}
            if dist.get_rank() == 0:
                _LOGGER.info(
                    "Colocated HCCL route: producer_rank=%s device=%s receiver_devices=%s world_size=%s group=%s",
                    owner, devices[owner], receivers, init["world_size"], group_id,
                )

            def join(selected_owner=owner, selected_init=init):
                return self._trainer_init(selected_init) if dist.get_rank() == selected_owner else None

            group, replies = _rpc_with_collective(
                client, endpoint, "init_colocated_packed_weight_group",
                {"group_id": group_id, **init, "receiver_devices": receivers,
                 "producer_device": devices[owner]}, join, timeout=180,
            )
            self._route_groups[group_id] = group
            self._validate_replies(replies, receivers, "joined")
            self._routes.append((owner, receivers, group_id))
        return endpoint

    def send_packed_bucket(
        self, client: Any, context: str, bucket_index: int, metadata: list[Mapping[str, Any]],
        total_bytes: int, packed: Optional[Any], policy_version: int,
    ) -> PackedWeightAck:
        """Stage one buffer per Trainer and send both disjoint receiver subsets in order."""
        if context != self._endpoint or len(self._routes) != 2:
            raise RuntimeError("Colocated HCCL routes are not initialized")

        def allocate():
            kind = (torch.accelerator.current_accelerator() or torch.device("cpu")).type
            device = torch.device("cpu") if kind == "cpu" else torch.device(
                kind, torch.get_device_module().current_device(),
            )
            if dist.get_rank() == 0:
                if packed is None or packed.numel() != total_bytes:
                    raise ValueError("Colocated HCCL producer buffer differs from metadata")
                return packed.to(device)
            return torch.empty(total_bytes, dtype=torch.uint8, device=device)

        local = synchronized_call("colocated HCCL allocation", allocate)
        try:
            dist.broadcast(local, src=0)
            synchronized_call("colocated HCCL staging complete", torch.get_device_module().synchronize)
            for owner, receivers, group_id in self._routes:
                def broadcast(selected_owner=owner, selected_group=group_id):
                    if dist.get_rank() == selected_owner:
                        self._route_groups[selected_group].broadcast(local, src=0)
                        torch.get_device_module().current_stream().synchronize()

                _, replies = _rpc_with_collective(
                    client, context, "receive_colocated_packed_weights",
                    {"group_id": group_id, "receiver_devices": receivers, "metadata": list(metadata),
                     "total_bytes": int(total_bytes), "policy_version": int(policy_version)},
                    broadcast, timeout=600,
                )
                self._validate_replies(replies, receivers, "received", total_bytes)
        except Exception:
            self._failed_buffers.append(local)
            raise
        return PackedWeightAck(bucket_index, total_bytes, len(self._physical_context.workers))

    def close(self) -> None:
        """Release route state only after rollout shutdown completes."""
        self._route_groups.clear()
        self._routes.clear()
        self._physical_context = None
        self._ownership.close()
        super().close()
