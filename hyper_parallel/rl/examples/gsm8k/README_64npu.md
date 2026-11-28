# Qwen3-30B-A3B GSM8K on four Ascend nodes

This recipe adapts the paired 16-NPU recipes from
[PR #1533](https://github.com/mindspore-ai/hyper-parallel/pull/1533)
to four nodes with 16 NPUs each. It uses the PR's current model replacements,
probability computation, trainer and full-gather weight publication.
The fork's earlier cluster entry and old unified operator bindings are replaced
by the PR implementation; use the new entry below rather than the old
`run_qwen3_moe_cluster.py` command.

Compared with the earlier A3 branch, training changes from TP4/DP8/EP4/CP2
to TP2/DP32/EP8/CP1. Full recomputation and rollout-time offload replace the
earlier disabled settings. The batched-token limit changes from 8192 to 2048,
and maximum sequences from 64 to 32. Learning rate, KL and weight decay retain
their existing `1e-6`, `0.001` and `0.01` values. The consistency attention now
uses the PR's grouped-attention factory instead of the older binding path.

## Changes from the PR recipe

| Setting | PR: one node, 16 NPUs | Four nodes, 64 NPUs |
| --- | --- | --- |
| Training TP / DP / EP / EDP / CP / SP | 2 / 8 / 8 / 2 / 1 / false | 2 / 32 / 8 / 8 / 1 / false |
| Questions per training DP rank | 16 | 4 |
| Global questions per step | 128 | 128 |
| Responses per question | 8 | 8 |
| Response mini-batch per DP rank | 128 | 32 |
| Training micro-batch | 2 | 2 |
| Rollout TP / DP / EP / PCP | 2 / 8 / 16 / 1 | Same, independently on each node |
| vLLM memory utilization | 0.8 | 0.4 |
| Maximum sequences per rollout replica | 128 | 32 |
| Maximum batched tokens | 2048 | 2048 |
| Evaluation batch per training DP rank | 64 | 16 |
| Final save | Resumable checkpoint and HF export | HF export only |

Both modes keep the PR's learning rate `1e-6`, KL coefficient `0.001`, weight
decay `0.01`, FP32 optimizer master parameters, full activation recomputation,
and `old_log_prob_source: actor`. Rollout log probabilities are collected for
diagnostics in both modes; the native run uses the recomputed Actor values as
the GRPO old-policy anchor. The native run does not enable numerical consistency.
Unified mode retains the PR's batch-invariant attention and expert execution.

Optimizer CPU offload and reference CPU offload remain disabled. The PR's
`offload_during_rollout: true` is enabled: Actor/reference residency is released
during generation and restored for training. This is different from permanently
offloading the optimizer or the reference role.

The graph mode is `FULL_DECODE_ONLY`; prompt and response limits are both 1024.
These defaults are a starting recipe, not a 64-NPU performance or stability
measurement. Run the two-step checks before the full experiments.

## Prepare each node

Use the same checkout, model and runtime on all four nodes. Follow the
[PR 30B recipe](README.md#launch-on-sixteen-devices) and
[consistency runtime instructions](../consistency/README.md): the prepared
environment needs vLLM/vLLM-Ascend **0.30** and the patched V4 numerical
consistency operators. The generic 0.23 image described in the Docker guide
does not establish compatibility with this 30B recipe.
Reinstall both editable packages after updating
the checkout. An older image from a previous run does not by itself prove runtime
compatibility.

```bash
python -m pip install --no-deps -e . -e hyper_parallel/rl
python -m pip install matplotlib pandas pyarrow pyyaml
```

Select four idle nodes before launching. This entry runs inside an already
prepared environment on every selected node; it does not discover hosts, create
Docker containers or clean up host containers. Each environment needs 16 visible
NPUs and working inter-node HCCL connectivity. Use one shared results filesystem
at the same absolute path on all nodes, including for the final collective save.

Start from the full official RL-format GSM8K splits. The sampling protocol follows
the [PR data preparation instructions](README.md#prepare-the-data). To reproduce
the earlier fork's fixed 6400/500 random sample with an index/hash manifest, run
once on CPU against the full, unsampled source:

```bash
python hyper_parallel/rl/examples/gsm8k/scripts/prepare_baseline_data.py --source /data/gsm8k-official --destination /data/gsm8k-6400-500
```

The script samples without replacement, restarting seed 42 separately for the
7473-question official training split and 1319-question official test split.
It records sampled indices and file hashes. The 500 test questions only participate
in evaluation. Reuse the same prepared files for both modes. The launch script
checks the row counts and records their hashes on each node.

## Launch

Run from the repository root on each selected node. Set the following variables
to identical values on all four nodes; set `NODE_RANK` individually to 0, 1, 2, 3.
Use a fresh `GSM8K_RUN_ID` for every experiment, including when changing its length.

```bash
export GSM8K_MODEL=/models/Qwen3-30B-A3B
export GSM8K_DATA=/data/gsm8k-6400-500
export GSM8K_RESULTS=/mnt/share/zjy/rl-cluster-runs
export MASTER_ADDR=172.27.3.101
export MASTER_PORT=29751
export GLOO_SOCKET_IFNAME=data0.172 HCCL_SOCKET_IFNAME=data0.172
export GSM8K_RUN_ID=qwen3-unified-2step
export NODE_RANK=0
bash hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_moe_64npu.sh unified 2
```

Use the actual data address of the selected rank-zero node for `MASTER_ADDR`.
For the native two-step check, use a fresh run ID and replace `unified 2` with
`native 2` on all four nodes. After both checks pass, run each 50-step experiment
with a new ID on all four nodes:

```bash
export GSM8K_RUN_ID=qwen3-unified-50step
bash hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_moe_64npu.sh unified 50
```

```bash
export GSM8K_RUN_ID=qwen3-native-50step
export GSM8K_COMPARE_WITH=/mnt/share/zjy/rl-cluster-runs/qwen3-unified-50step
bash hyper_parallel/rl/examples/gsm8k/scripts/run_qwen3_moe_64npu.sh native 50
```

Keep each node's own `NODE_RANK` unchanged. Optional configuration overrides use
the PR CLI syntax, for example `--rollout.vllm.gpu_memory_utilization=0.3`.
Training runs for exactly the supplied step count. Fifty steps use 6400 questions
at 128 per step, with one optimizer update per step; two steps use 256 questions.
Evaluation runs before training, every ten steps and at completion.

## Results and figures

For `RUN=$GSM8K_RESULTS/$GSM8K_RUN_ID`:

| Artifact | Location |
| --- | --- |
| Logs and resource samples | `$RUN/node0/run/train.log`, `$RUN/node*/run/resources.log` |
| Step and evaluation metrics | `$RUN/node0/run/steps.jsonl`, `evaluations.jsonl` |
| Completion report | `$RUN/summary.json` |
| Unified standalone figures | `$RUN/node0/run/plots/` |
| Native standalone figures | `$RUN/node0/run/native-plots/` |
| Paired figures, when reference supplied | `$RUN/node0/run/comparison-plots/` |
| Final 50-step model and tokenizer | `$RUN/checkpoints/step_50/hf/` |

Short checks do not save weights. Full runs export the final model without optimizer
or scheduler state; this HF directory supports inference, not exact training resume.
The source model directory is not overwritten. Preserve a full checkpoint instead
if exact resume is needed by setting `save_final_model_only=false` in the recipe.

The paired output contains four 6:4 figures in PNG and SVG: training reward (raw
and smoothed mean/std on one axis), 500-question test accuracy with final percentage,
mean absolute sampled-token train/rollout logp difference, and total runtime bars
for run total, rollout, Actor update and weight synchronization. No missing points
are extrapolated. Titles read the actual parallel settings from recorded configs;
rollout DP/EP describes each node's independent inference deployment.

Run total measures the rank-zero trainer lifecycle, including model construction,
evaluation and final export, but excludes CPU plotting and launcher exit. Component
bars sum all recorded training steps, including the first, and are not additive
with the run-total bar. Historical launcher totals can have a different scope;
the plotting report retains each run's recorded scope.

To regenerate a comparison without training:

```bash
python hyper_parallel/rl/examples/consistency/scripts/plot_comparison.py --consistency /mnt/share/zjy/rl-cluster-runs/qwen3-unified-50step --native /mnt/share/zjy/rl-cluster-runs/qwen3-native-50step --output /mnt/share/zjy/rl-cluster-runs/qwen3-native-50step/node0/run/comparison-plots
```
