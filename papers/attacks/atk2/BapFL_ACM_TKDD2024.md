# BapFL: You Can Backdoor Personalized Federated Learning

**期刊/年份**: ACM Transactions on Knowledge Discovery from Data (TKDD), Vol.18, No.7, Article 166, June 2024
**作者单位**: East China Normal University, Ant Group
**代码**: https://github.com/BapFL/code
**论文类型**: 攻击方法（针对参数解耦型PFL的后门攻击）

---

## 1. 核心问题与动机

- Qin et al. [24] (2023) 声称：带参数解耦（parameter decoupling）的 PFL 方法能有效抵御黑盒后门攻击。
- 本文反驳：这种"抵御能力"来自**恶意客户端与良性客户端之间分类器的异构性（heterogeneous classifiers）**，而非PFL方法本身的安全性。
- 两个直接原因导致分类器异构：
  - **F1**：客户端间天然存在数据异构性（class imbalance, concept-shift）
  - **F2**：恶意投毒进一步加剧了恶意/良性客户端间的数据分布差异（投毒改变了 p(y) 和 p(x|y)）

### 参数解耦类型
1. **Batch Normalization 解耦**：如 FedBN——仅BN层保留在本地，其余参数共享
2. **Feature Encoder + Classifier 解耦**：如 FedPer, FedRep, FedRod, FedBABU, FedMC——特征编码器 ω 共享，分类器 ψ 私有

本文主要针对 FedPer（因其训练过程简单），但分析可推广至其他方法。

---

## 2. 核心算法：BapFL

### 2.1 两大策略
- **PS1（Poison only feature encoder）**：只投毒特征编码器，保持分类器不变。这样恶意客户端的分类器优化始终基于干净数据，与良性客户端的分类器保持相近分布，规避 F2。
- **PS2（Diversify local classifier via noise）**：给攻击者的分类器 ψ_a 注入各向同性高斯噪声 N(0,σ)，模拟良性客户端分类器的多样性，从而使特征编码器学习到的后门能泛化到不同的（噪声化后的）分类器上，规避 F1。

### 2.2 优化目标
原始不可行目标（需要其他客户端分类器，攻击者无法获取）：
```
min_ω (1/N) Σ_i L̃_i(ω, ψ_i)
```
简化为攻击者自身可优化的形式：
```
min_ω (1/N) Σ_i L̃_a(ω, ψ_a + ε_i),  ε_i ~ N(0, σ)
```

### 2.3 算法流程（Algorithm 1: Training of BapFL）
```
输入：初始特征编码器ω0，各客户端分类器{ψi}，总轮数R，本地迭代次数τ，
      batch size B，学习率η，高斯噪声方差σ，每轮参与客户端数M，
      每batch投毒样本数b

for round r = 0..R-1:
    选取客户端集合 S_r（大小M）
    for 每个客户端 ci ∈ S_r:
        ωi ← ωr (下载全局特征编码器)
        将本地数据集切分为batch
        for 每次迭代 t = 1..τ:
            采样一个batch (Xt, Yt)
            if ci 是攻击者:
                随机投毒b个样本 → (Xt_clean,Yt_clean), (Xt_poison,Yt_poison)
                ωi ← ωi - η∇ωi Li(Xt_clean, Yt_clean)   # 用干净数据更新特征编码器
                ψi ← ψi - η∇ψi Li(Xt_clean, Yt_clean)   # 用干净数据更新分类器
                ε ← 从N(0,σ)采样噪声
                ψi ← ψi + ε                              # PS2: 扰动分类器
                ωi ← ωi - η∇ωi Li(Xt_poison, Yt_poison)  # PS1: 只用投毒数据更新特征编码器
            else:
                正常更新 ωi, ψi
        ω_{r+1,i} ← ωi
        客户端将 ω_{r+1,i} 发回服务器
    # 服务器聚合（仅聚合特征编码器ω，按数据量加权FedAvg）
    ω_{r+1} ← Σ_ci∈Sr (|Di| / Σ|Di|) * ω_{r+1,i}
```

---

## 3. 实验设置

### 3.1 数据集与模型
| 数据集 | 模型 |
|---|---|
| MNIST | 4层ConvNet（2个卷积层+2个全连接层） |
| Fashion-MNIST | 同上ConvNet |
| CIFAR-10 | VGG11（8个卷积层+3个全连接层） |

- 参数共享层数：ConvNet共享前3层；VGG11共享前6层
- Non-IID划分：50个客户端，Dirichlet分布 α=0.5

### 3.2 训练超参数
| 参数 | 取值 |
|---|---|
| 通信轮数 | MNIST 200 / Fashion-MNIST 400 / CIFAR-10 1000 |
| 本地迭代次数 τ | 20 |
| 本地batch size B | 64 |
| 学习率 η | 0.1，每10轮衰减0.99 |
| 优化器 | SGD，weight decay = 1e-4 |
| 每轮参与客户端比例 | 10%（5个客户端） |

### 3.3 攻击设置
- 触发器：grid pattern（网格图案，见Fig.4）
- 攻击策略：all-to-one，所有中毒样本目标标签统一为 class 2
- 恶意客户端数：随机指定 2 个
- 攻击开始时机：主任务准确率收敛后开始（MNIST: round 50；Fashion-MNIST: round 200；CIFAR-10: round 500），此后每轮持续攻击
- 每batch投毒样本数：MNIST/Fashion-MNIST = 20个；CIFAR-10 = 5个
- **BapFL 的 σ 搜索范围**：{0.01, 0.05, 0.1, 0.15, 0.2}；默认 MNIST/Fashion-MNIST σ=0.2，CIFAR-10 σ=0.01

### 3.4 Baseline 攻击
- **Black-box Attack** [24]：同时更新特征编码器和分类器，最小化干净+投毒样本的联合损失
- **Scaling Attack** [2]：模型更新缩放因子 = 5
- **DBA** [32]：全局触发器拆分为多子触发器分布给多个恶意客户端

### 3.5 评价指标
- **ASR**（Attack Success Rate）：投毒测试数据上的分类准确率
- **MTA**（Main-Task Accuracy）：干净测试数据上的分类准确率
- 评估频率：MNIST每2轮，Fashion-MNIST每4轮，CIFAR-10每10轮；每个实验5次随机种子重复，取最后50轮的平均结果

---

## 4. 主要实验结果

### 4.1 主结果（Section 4.2）
- BapFL 显著优于所有基线，且不损害 MTA。
- BapFL 相比最佳基线 Scaling Attack：MNIST 高约 **57%**，Fashion-MNIST 高约 **42%**，CIFAR-10 高约 **36%**
- 具体ASR（摘要中提到）：BapFL 达到 **94.21%（MNIST）、80.93%（Fashion-MNIST）、58.89%（CIFAR-10）**

### 4.2 数据异构性影响（Section 4.3, Fig.7）
- 控制 Dirichlet α ∈ {0.1, 0.3, 0.5, 1, 3}（α越小异构性越强）
- α=0.1（强异构）：BapFL ASR = 46.82%(MNIST) / 35.58%(F-MNIST) / 18.25%(CIFAR-10)，仍超过最佳基线约20%/10%/12%
- α=3（弱异构）：BapFL ASR = 99.96%(MNIST) / 99.93%(F-MNIST) / 68.27%(CIFAR-10)；对比最佳基线仅 55.10%/40.77%/15.59%

### 4.3 共享层数量影响（Section 4.4, Fig.8, CIFAR-10）
- L ∈ {4,5,6,7,8,9,10}（共享层数）
- 共享层越多，越易受攻击；共享层太少则损失模型效用
- L=6 是较好平衡点：相比 L=7 可将基线攻击ASR降低超40%，但MTA仅损失约1%
- **即使 L=6，BapFL 仍能达到 58.89% ASR**

### 4.4 恶意客户端数量（Section 4.5, Fig.9）
- A ∈ {1,2,3,4,5}
- 仅1个攻击者时：BapFL ASR = 89.67%(MNIST) / 73.72%(F-MNIST) / 53.52%(CIFAR-10)
- 5个攻击者时 Scaling Attack 也仅达到 56.78%/66.20%/48.04%

### 4.5 攻击频率（Section 4.6, Fig.10）
- 攻击间隔 I ∈ {1,2,4,8,16}
- BapFL 在 I≤8 时仍有效；I=16 时性能显著下降
- 结合 Scaling Attack（BapFL+Scaling）可显著提升可持续性：I=16 时仍保持 ASR 83.64%(MNIST)/70.17%(F-MNIST)/46.25%(CIFAR-10)

### 4.6 消融实验（Section 4.7, Fig.11）
- σ=0 时（仅PS1，记为 BapFL⁻）：ASR = 78.46%(MNIST)/71.52%(F-MNIST)/49.3%(CIFAR-10)，仍超过最佳基线约41%/34%/27%
- 加入PS2（最优σ）后：相比 BapFL⁻ 再提升约15%(MNIST) / 9%(F-MNIST, CIFAR-10)
- σ搜索范围：MNIST/F-MNIST: 0.04×{0,1,...,6}；CIFAR-10: 0.002×{0,1,...,6}
- σ过大会导致优化不稳定，攻击性能反而下降

### 4.7 防御评估（Section 5, Table 1，MNIST α=0.5）
六种防御方法：Gradient Norm-clipping (H∈{0.1,0.3,0.5})，Median，Trimmed Mean (β∈{0.2,0.4})，Multi-Krum (f=2, J=3)，Fine-Tuning（20轮迭代），Simple-Tuning（重新初始化分类器后训练200轮迭代）

| 防御 | No-defense | Grad-clip(H=0.5) | Median | Trimmed(β=0.4) | Multi-Krum | Fine-tuning | Simple-tuning |
|---|---|---|---|---|---|---|---|
| BapFL ASR | 94.21 | 88.52 | 94.20 | 52.91 | 9.18 (需PGD绕过) | 94.63 | 88.65 |

- **仅 Multi-Krum 提供有效防御**（ASR降至9.18%），但结合 **PGD攻击（δ=0.01，projected gradient descent）** 后 BapFL 可绕过 Multi-Krum，达到 **ASR = 70.60%**
- Fine-Tuning 几乎无效，甚至可能提高ASR
- 提出组合防御 **MKST**（Multi-Krum + Simple-Tuning）：可将 BapFL 的 ASR（结合PGD后）从70.60%进一步降至68.30%，但仍然显著

---

## 5. 面向 Claude Code 复现的关键要点摘要

1. **核心创新是两个简单策略的组合（PS1+PS2）**，实现复杂度低，计算/存储开销几乎为零。
2. **关键实现点**：
   - 攻击者训练循环中，特征编码器的梯度来自**投毒数据**，分类器的梯度只来自**干净数据**（PS1）
   - 每次迭代都要在分类器参数上叠加一次高斯噪声 N(0,σ)（PS2），注意噪声是"注入"而非"替换"（`ψi ← ψi + ε`）
3. 触发器本身很简单（grid pattern，见Fig.4），复现重点在训练流程而非触发器设计（这点与 Bad-PFL 相反）
4. 建议先用 FedPer 复现（默认目标方法），后续可扩展到 FedRep/FedRod/FedBABU/FedMC
5. 若要测试防御鲁棒性，需要实现 PGD 投影步骤：`ω' ← ω' 投影到以 ω° 为中心、半径 δ=0.01 的球内`（用于绕过 Multi-Krum）
6. 关键超参数速查：50 clients, Dirichlet α=0.5，本地迭代τ=20，batch=64，lr=0.1（每10轮×0.99衰减），2个恶意客户端，每轮5个客户端参与
