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
"""Portable topology, metrics and failure-report regressions without devices."""
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd
import yaml

from hyper_parallel.rl.examples.consistency.scripts import configure_run as configure
from hyper_parallel.rl.examples.consistency.scripts import prepare_gsm8k as data
from hyper_parallel.rl.examples.consistency.scripts import run_consistency as runner

ROOT = Path(configure.__file__).resolve().parent

def valid_metrics():
    """Return finite independent probability evidence for a nonempty action."""
    return {f'training/{key}': value for key, value in {
        'rollout_probs_diff_valid': 1, 'log_probs_valid_tokens': 32,
        'rollout_log_probs_diff_max': 0, 'rollout_log_probs_diff_mean': 0,
        'actor_log_probs_mean': -2, 'rollout_log_probs_mean': -2}.items()}


class ReportingTests(unittest.TestCase):
    """Keep portable recipes and observation reports faithful to the runtime."""

    def test_gate_rejects_missing_nonfinite_empty_and_nonzero(self):
        """Never relax the zero-error gate for missing or invalid evidence."""
        runner.check_metrics(valid_metrics())
        for key, value in [('log_probs_valid_tokens', 0), ('rollout_probs_diff_valid', 0),
                           ('rollout_log_probs_diff_max', 1e-7), ('actor_log_probs_mean', float('nan'))]:
            metrics = valid_metrics()
            metrics[f'training/{key}'] = value
            with self.assertRaises(RuntimeError):
                runner.check_metrics(metrics)
        with self.assertRaises(KeyError):
            runner.check_metrics({})

    def test_csv_preserves_metrics_and_does_not_invent_missing_fields(self):
        """Persist the original measurements and retain absent metric semantics."""
        with tempfile.TemporaryDirectory() as directory:
            tracker = types.SimpleNamespace(log=lambda *args, **kwargs: None)
            observer = runner.MetricsObserver(tracker, Path(directory))
            observer.log({'train/gradient_norm': 1.25}, step=1)
            observer.log({'rollout/reward_mean': 0.5}, step=2)
            rows = [json.loads(line) for line in Path(directory, 'steps.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['train/gradient_norm'], 1.25)
            self.assertNotIn('train/gradient_norm', rows[1])
            self.assertIn('rollout/reward_mean', Path(directory, 'steps.csv').read_text())

    def test_three_steps_and_original_tracker_are_preserved(self):
        """Allow arbitrary run lengths while retaining every pre-update gate."""
        state = types.SimpleNamespace(global_step=0)
        targets = types.SimpleNamespace(prepare=lambda *args: (None, valid_metrics()))
        original = types.SimpleNamespace(log=lambda *args, **kwargs: None)
        trainer = types.SimpleNamespace(state=state, targets=targets, _tracker=original)
        trainer.rollout_manager = types.SimpleNamespace(generate=lambda *args: None)
        trainer.actor = types.SimpleNamespace(update=lambda *args: None)
        trainer.publication = types.SimpleNamespace(publish=lambda *args: None)
        def train():
            for step in range(1, 4):
                targets.prepare(types.SimpleNamespace(rollout_log_probs=[-2]), True, {})
                state.global_step = step
                trainer._tracker.log({'train/gradient_norm': step}, step=step)
        trainer.train = train
        config = {'train': {'max_steps': 3, 'accelerator': {}}, 'logging': {'log_steps': 1},
                  'rollout': {'vllm': {}}}
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, 'build_trainer', return_value=trainer), \
             patch.object(runner, 'run_synchronized', side_effect=lambda name, callback: callback()):
            destination = Path(directory, 'consistency.json')
            runner.run(config, destination)
            result = json.loads(destination.read_text())
            self.assertEqual(result['completed_steps'], 3)
            self.assertEqual(len(result['checks']), 3)
            self.assertIs(trainer._tracker.tracker, original)

    def test_main_failure_persists_summary_and_reraises(self):
        """Reject stale success reports and preserve the original failure."""
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory, 'consistency.json')
            report.write_text('{"passed": true}')
            config = {'train': {'max_steps': 2, 'accelerator': {}}, 'rollout': {'vllm': {}}}
            with patch.object(sys, 'argv', ['run_consistency.py', 'unused.yaml', '--report', str(report)]), \
                 patch.object(runner, 'load_config', return_value=config), \
                 patch.object(runner, 'run', side_effect=RuntimeError('injected startup failure')):
                with self.assertRaises(RuntimeError):
                    runner.main()
            summary = json.loads(report.with_name('summary.json').read_text())
            self.assertFalse(summary['passed'])
            self.assertEqual(summary['completed_steps'], 0)
            self.assertIn('injected startup failure', summary['error'])
            self.assertFalse(report.exists())

    def test_topology_generation_and_invalid_divisibility(self):
        """Derive sixteen-device grids and reject impossible device counts."""

        path = ROOT.parent / 'configs/qwen3_moe_4l_rl.yaml'
        with patch.dict(os.environ, {'TRAIN_TP': '4', 'TRAIN_EP': '4', 'ROLLOUT_TP': '2',
                                    'ROLLOUT_EP': '8', 'STEPS': '3', 'RUN_MODEL_PATH': '/models/full'}, clear=True):
            config = configure.configure(yaml.safe_load(path.read_text()), 16)
            self.assertEqual(config['train']['accelerator']['dp_shard'], 4)
            self.assertEqual(config['rollout']['vllm']['data_parallel_size'], 8)
            self.assertEqual(config['train']['max_steps'], 3)
            with self.assertRaises(ValueError):
                configure.configure(yaml.safe_load(path.read_text()), 3)

    def test_data_answers_are_not_in_prompts_and_overwrite_is_rejected(self):
        """Keep reference answers private and protect original prepared files."""

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, 'raw')
            source.mkdir()
            for split in ('train', 'test'):
                (source / f'{split}.jsonl').write_text(json.dumps({'question': 'Calculate 2+2',
                                                                  'answer': 'private reasoning #### 4'}) + '\n')
            output = Path(directory, 'prepared')
            data.convert(source, output)
            frame = pd.read_parquet(output / 'train.parquet')
            self.assertEqual(frame.iloc[0]['prompt'], 'Calculate 2+2')
            self.assertEqual(frame.iloc[0]['extra_info']['answer'], '4')
            with self.assertRaises(ValueError):
                data.convert(source, output)


if __name__ == '__main__':
    unittest.main(verbosity=2)
