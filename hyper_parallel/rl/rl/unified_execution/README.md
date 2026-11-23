# Unified Execution

HyperParallel 与 vLLM 复用模型、参数语义和关键数值计算，各自保留训练和推理运行时。
接入新模型时，从[新模型训推一致性接入与生产验收](../../docs/unified_execution_model_onboarding.md)开始。
当前 Qwen3 实现说明集中在两篇文档：

- [统一执行核心流程](../../docs/unified_execution_guide.md)：模型加载与替换、执行契约、Level 3、PIK、
  Attention、CP/PCP、权重切分与发布，以及支持范围。
- [MoE 与 EP 实现](../../docs/unified_execution_moe.md)：数值合并规则、训练动态路由、推理 vLLM-Ascend 原生专家与源行交换、
  图执行、容量和专家所有权。

[运行两份配置及不同拓扑的验证矩阵](../../examples/consistency/README.md)。

## 代码导航

| 目录 | 职责 |
| --- | --- |
| [core/](core/) | 执行契约、显式配方、进程数值策略 |
| [models/](models/) | 模型组成与共享 PIK/专家/router 绑定 |
| [ops/](ops/) | V4 Attention、PIK、训练 EP 与共享数值合并 |
| [runtimes/hyper_parallel/](runtimes/hyper_parallel/) | HP 构建与结构校验、训练 Attention/CP、可微 EP 接入 |
| [runtimes/vllm/](runtimes/vllm/) | 推理协议、并行域、MoE 图执行、分页 Attention、权重加载 |

共用绑定不导入运行时。模型完整 forward、参数切分与布局仍由 HP 负责；调度、KV cache 和图回放
仍由 vLLM 负责。训练与推理复用 HP 结构校验，各自核对绑定到模型的 EP 实现。
