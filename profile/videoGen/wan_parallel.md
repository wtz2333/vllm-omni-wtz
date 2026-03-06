非常感谢您的指正。在 VideoGen（特别是 Wan2.2 这种 14B MoE 模型）的推理中，并行配置确实非常灵活且存在严格的约束。

根据 vLLM-Omni 的最新文档和代码逻辑（如 PR #1339 引入的 HSDP 以及 PR #966 等并行特性），推理并行的核心逻辑遵循乘积规则，且 **HSDP 与 TP 互斥**。

以下是为您整理的 2/4/8 张显卡下所有可能的并行组合及其配置逻辑：

### 核心配置原则

1. **基础公式**：$TP \times SP \times CFG = N$（总显卡数）。


2. **SP 细分**：$SP = Ulysses \times Ring$。
3. **HSDP 约束**：HSDP（混合分片数据并行）利用 PyTorch FSDP2 进行权重分片，它 **不能** 与张量并行（TP）同时开启（即使用 HSDP 时，`tensor_parallel_size` 必须为 1）。


4. **HSDP 逻辑**：$hsdp\_shard\_size \times hsdp\_replicate\_size = N$ 。



---

### 1. 2 张显卡时的组合 ($N=2$)

| 类别 | 并行组合 ($TP, SP, CFG$) | HSDP 配置 | 说明 |
| --- | --- | --- | --- |
| **纯 TP** | (2, 1, 1) | 无 | 最基础的显存分担方式。 |
| **纯 SP** | (1, 2, 1) | 无 | 针对长视频序列，减少 Attention 显存。 |
| **纯 CFG** | (1, 1, 2) | 无 | 侧重速度，有条件/无条件分支并行。 |
| **HSDP** | (1, 1, 1) | `shard=2, rep=1` | **独立模式**：通过 FSDP2 分片权重，不使用其他并行。 |

---

### 2. 4 张显卡时的组合 ($N=4$)

四卡开始支持 HSDP 的“结合模式”（Combined Mode），即将 HSDP 覆盖在 SP 或 CFG 之上 。

| 类别 | 并行组合 ($TP, SP, CFG$) | HSDP 配置 | 说明 |
| --- | --- | --- | --- |
| **纯并行** | (4, 1, 1) 或 (1, 4, 1) | 无 | 纯 TP4 或纯 SP4（Ulysses/Ring）。 |
| **TP + SP** | (2, 2, 1) | 无 | **推荐组合**：兼顾权重分片与长序列处理。 |
| **TP + CFG** | (2, 1, 2) | 无 | 兼顾显存缩减与推理加速。 |
| **SP + CFG** | (1, 2, 2) | 无 | 纯序列与引导并行。 |
| **HSDP 独立** | (1, 1, 1) | `shard=4, rep=1` | 4 路权重分片，无其他并行。 |
| **HSDP 结合** | (1, 2, 1) | `shard=2, rep=2` | **HSDP + SP**：2 路 SP，每路 SP 内部再做 2 路 HSDP 分片。 |
| **HSDP 结合** | (1, 1, 2) | `shard=2, rep=2` | **HSDP + CFG**：CFG 两路各占 2 卡，内部做 HSDP 分片。 |

---

### 3. 8 张显卡时的组合 ($N=8$)

八卡环境下，HSDP 的作用更加明显，可以有效解决 14B MoE 模型在长序列下的显存爆炸问题 。

| 类别 | 并行组合 ($TP, SP, CFG$) | HSDP 配置 | 说明 |
| --- | --- | --- | --- |
| **高阶 TP/SP** | (8, 1, 1) 或 (1, 8, 1) | 无 | 纯 TP8 或纯 SP8。 |
| **TP + SP** | (4, 2, 1) 或 (2, 4, 1) | 无 | 常用于处理超长视频（如 81 帧以上）。 |
| **TP + CFG** | (4, 1, 2) | 无 | **高性能**：TP4 解决显存，CFG2 翻倍速度。 |
| **SP + CFG** | (1, 4, 2) | 无 | 针对长序列的快速生成。 |
| **全维度混合** | (2, 2, 2) | 无 | **全能型**：各维度均摊，适合 81 帧/480p 标准任务。 |
| **HSDP 独立** | (1, 1, 1) | `shard=8, rep=1` | 8 路权重全分片。 |
| **HSDP + SP** | (1, 4, 1) | `shard=2, rep=4` | 4 路 SP，每路内部 2 路 HSDP 分片。 |
| **HSDP + SP** | (1, 2, 1) | `shard=4, rep=2` | 2 路 SP，每路内部 4 路 HSDP 分片。 |
| **HSDP + CFG** | (1, 1, 2) | `shard=4, rep=2` | CFG 两路各占 4 卡，内部 4 路 HSDP 分片。 |
| **HSDP 混合** | (1, 2, 2) | `shard=2, rep=4` | **最高级组合**：2路SP $\times$ 2路CFG $\times$ 2路HSDP。 |

---

### 关键特性说明与用法：

* **HSDP 的用法**：
在 `DiffusionParallelConfig` 中设置：
```python
parallel_config = DiffusionParallelConfig(
    use_hsdp=True,
    hsdp_shard_size=4,   # 权重分片份数
    hsdp_replicate_size=2, # 副本组数
    ulysses_degree=2      # 结合 SP 使用
)

```


*注意：此时 `tensor_parallel_size` 必须默认为 1* 。


* **SP 的混合用法**：
在 4/8 卡时，SP 本身可以由 Ulysses 和 Ring 组合：
例如 $SP=4$ 可以配置为 `ulysses_degree=2, ring_degree=2`。这在跨机柜（Node）部署时非常有用，通常在节点内用 Ulysses，节点间用 Ring 。



