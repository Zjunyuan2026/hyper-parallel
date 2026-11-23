# Unified Execution：MoE 数值一致性与 EP 实现

MoE 两端共用 HP 的 router、专家 ID 和已发布权重的逻辑布局。训练使用可微动态路由。推理 EP 使用一套 vLLM-Ascend 原生路由和专家 FFN。并行拓扑决定 token 在哪些 rank 上交换。通信和路由算子可以不同，但必须保留每条路由的数值含义和最终合并规则。
完整模型构建、Attention、CP/PCP 与发布流程见 [统一执行说明](unified_execution_guide.md)。

新模型的接入顺序、组件选型和生产验收见[新模型训推一致性接入与生产验收](unified_execution_model_onboarding.md)。

## 数值契约

设本 rank 有 T 个 token、每 token 选择 K 个专家、hidden size 为 H、专家总数为 E，EP 度为 P：

| 张量 | shape | 含义 |
| --- | --- | --- |
| `states` | `[T,H]` | 进入 MoE 的本地 token 行 |
| `indices` | `[T,K]` | 原始 top-k 槽位顺序中的全局专家 ID |
| `weights` | `[T,K]` | router 输出的原始分数（Qwen3 BF16 模型会将归一化分数转回 BF16） |
| `routes` | `[T,K,H]` | 每个槽位对应的未加权专家输出 |
| 输出 | `[T,H]` | 按固定顺序合并后的结果 |

每条 `(source rank, token, top-k slot)` 路由恰好计算一次；不丢弃拥塞专家的 token。
专家使用 HP `GroupedExperts` 的 EIO 权重布局、SwiGLU 和 grouped GEMM。
专家输出回到源 token 后，遵循 [ordered_combine()](../rl/unified_execution/ops/moe/combine.py) 的顺序：

```python
result = torch.zeros_like(routes[:, 0], dtype=torch.float32)
for slot in range(routes.shape[1]):
    product = routes[:, slot].float() * weights[:, slot, None].float()
    result = result + product
return result.to(routes.dtype)
```

不能提前把 router 权重转成 BF16，不能把各 owner 的局部加权和用 AllReduce 合并，
也不能未经验证把槽位循环替换成 `sum`/`index_add_`。这些操作在数学上等价，但可能改变舍入顺序。
推理保留每条路由的未加权输出，恢复到源 rank 后再以原始分数合并。

## 绑定和执行边界

[models/bindings.py](../rl/unified_execution/models/bindings.py) 的 `bind_expert_ownership()`
复用 HP 的本地专家绑定和模型族 router，返回 EP group、rank、world size 和本地专家数。

| 角色 | 入口 | 执行路径 |
| --- | --- | --- |
| 训练 | [unified_ep_compute_fn()](../rl/unified_execution/runtimes/hyper_parallel/moe.py) | router → `training_ep` → 固定顺序合并 |
| 推理 | [inference_ep_compute_fn()](../rl/unified_execution/runtimes/vllm/moe.py) | router → 注册的 `InferenceExpertExecutor` → vLLM-Ascend 路由、专家 FFN 和源行合并 |

角色在模型构建时确定；Actor 调用 `eval()` 计算概率不会切换为推理通信。
worker builder 从共享配方生成内部推理绑定，保留原 Actor 配方和指纹。
`validate_training_ep()` / `validate_inference_ep()` 检查实际绑定的计算函数、router、专家归属、通信组和图注册对象，
而非只检查模块名或一张静态配置表。

## 专家域与权重所有权

训练继续使用 HP 的专家 mesh；CP 可以参与专家域，不另设一套专家分片。
推理在原生 `DP × PCP × TP` 的 rank 顺序上构建 HP mesh，EP 度必须整除该域：

```text
DP1 × PCP2 × TP2：rank 顺序 [0,1,2,3]
EP2 → [0,1] 与 [2,3]，每组各有完整专家集合的两个分片
EP4 → [0,1,2,3]，专家分片跨两个 PCP 平面
```

较小的 EP 自然落在 TP 内，较大的 EP 跨 PCP 或 DP；没有额外的“跨 PCP”开关。
PCP1 时退化为 DP×TP。推理把 PCP 折入 HP 的非 TP 构建轴，但 HP 模型 CP 仍为 1，
不改变 vLLM 调度器的 DP/PCP 语义。

一个 EP rank 拥有连续的 E/P 个专家。
[parallel.py](../rl/unified_execution/runtimes/vllm/parallel.py) 负责通信域，
[LogicalWeightLoader](../rl/unified_execution/runtimes/vllm/weights.py) 按同一布局解释权重。
逻辑 `experts.e.gate_proj/up_proj/down_proj` 转换到本地 EIO 的 gate/up 与 down 区域，
只复制本 rank 所有的专家。发布接收端覆盖 DP×PCP 副本坐标，并保留实际 TP 宽度。

更新使用原址 `copy_`，不替换 Parameter；图中的地址保持有效。EP1 的 `when: ep` 规则不生效，
继续原本的本地专家路径。专家所有权由 HP 管理，也没有专家内部 ETP/PIK。

## 训练：动态 AllToAll 与可微还原

[training_ep.py](../rl/unified_execution/ops/moe/training_ep.py) 复用普通训练 EP 的
`_prepare_ep_dispatch()` 和 `_run_ep_local_experts()`：

```text
states / 原始 IDs / FP32 weights
    → 动态路由与变长 AllToAll
    → module.experts(...)：保留模块调用、FSDP hooks 和 autograd
    → 逆 AllToAll 返回路由结果
    → dispatch_order 的逆排列，恢复 [T,K,H]
    → ordered_combine(routes, 原始 FP32 weights)
```

逆通信后仍是 dispatch 排列，必须先还原槽位，才能按相同顺序合并。
这里不使用普通 dispatch 为自身计算准备的低精度权重副本。
反向检查输入、router 权重和专家参数的梯度正确性，
不要求跨 EP/CP 的梯度逐位相同。

## 推理端的原生专家计算

每个 worker 创建一个模型级的 [NativeQwen3Experts](../rl/unified_execution/runtimes/vllm/native_moe.py)。每层沿用自己的 router 和 HP 分片。初始化、在线发布和下一次 forward 均使用同一组参数地址。

1. **确定行归属。** 在一个 HP EP 组内，同一 TP 平面且 SP 关闭的 lane 共享 token 行。每个 lane 只与其他平面的对应 lane 交换不同的源行。SP 开启时每个 lane 都是独立来源。EP 子组独立处理自己的 token 和全局编号的专家。
2. **收集输入。** 多来源组对齐实际调度行数，将 BF16 隐状态和 int32 专家 ID 的原始位装入一条 AllGather 消息。ID 不会因低精度舍入改变。padding 使用超出真实专家范围的哨兵。整个 EP 组都拥有同一份行时省略这次交换。router 权重留在源 rank。
3. **执行本地专家。** 调用 vLLM-Ascend 的 `DeviceOperator.npu_moe_init_routing`，显式传入当前 HP 分片的专家区间。将路由计数和 HP 的 EIO 权重交给 `apply_moe_mlp`。这里复用原生路由和专家计算内核，worker 仍负责 HP 与 vLLM 的并行布局适配。
4. **归还并合并。** [槽位还原内核](../rl/unified_execution/ops/moe/restore_slots.py) 在一次访存中判断专家所有权、按逆排列读出真实贡献、为非 owner 填零。每个 rank 拥有不同源行时，EP reduce-scatter 将槽位送回各自的源 rank。组内有 TP 副本时，EP all-reduce 拼齐槽位，各 rank 取回自己的源行。不同 DP 副本按公共物理行数通信，再裁回各自的实际行数。BF16 分数、top-8 形状使用预置正零槽的原生 unpermute，其余分数精度和形状使用共享的顺序合并函数。

例如 TP2、PCP2、EP4 且关闭 SP 时，rank 0 与 1 共享一组 token，2 与 3 共享另一组。0 与 2、1 与 3 分别交换源行。四个 rank 用一次 EP all-reduce 拼齐未加权槽位，0 与 1 取第一份源行，2 与 3 取第二份。开启 SP 后，四个 rank 各自代表一个来源，reduce-scatter 直接把所需的行送回各自的 rank。

## 图执行的职责划分

`execute_inference_ep` 是 `hyper_rl::inference_ep` 自定义算子：Dynamo 只看到张量接口和 fake shape，
原生执行/捕获时再读取调度上下文，访问模型所属的 `static_forward_context` 中的 executor。
执行时直接访问已绑定的专家权重，通信和 GMM 可进入 ACL 捕获。
实际图回放使用该图已经捕获的路径，匹配的图由 vLLM runner 选择。

HP 模型构造完成后，`register_inference_experts()` 按完整模块路径注册 executor，例如
`hyper_rl.ep.model.layers.0.mlp`；存在 vLLM 模型前缀时也将其纳入键名。
该键会作为常量保存在 AOT 图中，因此不能使用注册序号：HP 边界安装顺序可能随进程变化，
重启后复用序号会静默调用其他层的同形状专家。注册必须在结构审计与编译之前完成，重复键直接报错。

eager 和图模式共用同一套专家计算。
图回放主要减少主机发射开销；未被图覆盖的 prefill 或尺寸仍由原生 runner 执行。
训练动态通信与推理图通信可以不同，数值契约不能因此放宽。

## 验证与性能边界

[EP 设备测试](../tests/st/test_unified_expert_parallel.py) 验证路由、逆排列、空 rank、拥塞专家、
有效计数变化、原址权重更新及动态图回放，并检查训练梯度。
[PCP 测试](../tests/st/test_prefill_context_parallel.py) 另外检查跨 PCP 的专家归属与发布。
[两份配置的运行矩阵](../examples/consistency/README.md#配置与验证矩阵) 检查完整模型实际采样概率。
组件图捕获成功不能替代完整模型的 sleep/wake、非零更新及第二步发布。
缓存复用还需在同一缓存目录连续启动独立进程，确认第二次实际加载 AOT 产物，且更新前后的
采样概率均与 Actor 相同；每次清空缓存不能覆盖跨进程的层绑定正确性。

性能对照应分别测量短 decode、长 prefill 和长短混批，并记录源 rank 数量、TP 副本数及有效行比例。不同来源越多，未加权槽的归还流量越大。单机测量仅代表实测拓扑和负载。
