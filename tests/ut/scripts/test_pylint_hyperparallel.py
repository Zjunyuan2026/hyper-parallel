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
"""Backend import checks follow the native Torch RL boundary."""
import astroid
from pylint.testutils import CheckerTestCase, MessageTest

from scripts.pylint_hyperparallel import HyperParallelChecker


class TestRLBackendImports(CheckerTestCase):
    """Exercise both import syntaxes inside and outside the RL runtime."""

    CHECKER_CLASS = HyperParallelChecker

    def test_rl_torch_imports_are_allowed(self):
        for path in ("/repo/hyper_parallel/rl/rl/trainer.py",
                     r"C:\repo\hyper_parallel\rl\rl\algorithm\rollout_correction.py"):
            self.checker._module_path = path
            for source in ("import torch", "import torch.distributed as dist", "from torch import Tensor"):
                node = astroid.parse(source).body[0]
                with self.assertNoMessages():
                    if isinstance(node, astroid.nodes.Import):
                        self.checker.visit_import(node)
                    else:
                        self.checker.visit_importfrom(node)

    def test_other_library_directories_remain_restricted(self):
        for path in ("/repo/hyper_parallel/utils/example.py",
                     "/repo/hyper_parallel/rl/rl_extra/example.py"):
            self.checker._module_path = path
            for source in ("import torch.distributed", "from torch.distributed import get_rank"):
                node = astroid.parse(source).body[0]
                with self.assertAddsMessages(MessageTest(
                        msg_id="forbidden-backend-import", node=node, args=("torch.distributed",)),
                        ignore_position=True):
                    if isinstance(node, astroid.nodes.Import):
                        self.checker.visit_import(node)
                    else:
                        self.checker.visit_importfrom(node)

    def test_existing_core_allowance_is_preserved(self):
        self.checker._module_path = "/repo/hyper_parallel/core/fully_shard/example.py"
        node = astroid.parse("import torch").body[0]
        with self.assertNoMessages():
            self.checker.visit_import(node)
