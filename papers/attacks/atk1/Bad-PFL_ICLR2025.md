# Bad-PFL: Exploring Backdoor Attacks against Personalized Federated Learning

**会议/年份**: ICLR 2025
**作者单位**: East China Normal University, Deakin University
**论文类型**: 攻击方法（针对个性化联邦学习的后门攻击）

---

## 1. 核心问题与动机

- 已有研究（Qin et al., 2023）声称：partial model-sharing 类型的 PFL 方法能显著抵御现有的联邦后门攻击（ASR 可降到 30% 以下）。
- 本文反驳该结论：现有攻击失败并非因为 PFL 天然免疫，而是因为**现有攻击使用人工设计的触发器（trigger），这种触发器难以在个性化模型中存活**。
- 识别出后门攻击在 PFL 中失败的**三个关键原因**：
  1. **Full model-sharing 方法**：仅靠正则化项（regularization term）不足以将后门从全局模型转移到个性化模型（除非对正则化项按梯度幅度加权）。
  2. **Partial model-sharing 方法**：非共享参数（如私有分类头、BN 层）不会自适应后门，从而阻断后门效果。
  3. **训练过程中的稀释效应**：（a）恶意客户端被服务器选中的间隔越长，全局模型中的后门越容易被良性更新稀释；（b）FL 结束后客户端在本地干净数据上微调，会进一步冲淡后门。

---

## 2. 核心算法：Bad-PFL

### 2.1 基本思想
利用目标类别的**自然特征（natural features）**作为触发器，而不是人工设计的图案。因为良性客户端的数据中天然包含目标类别的样本，其个性化模型训练时会不可避免地学到"自然特征 → 目标标签"的映射，从而隐式嵌入后门。该关系在个性化模型微调/训练过程中不会被遗忘（因为遗忘会导致目标类准确率下降）。

### 2.2 触发器构成
触发器 `T(x) = δ + ξ`，即目标特征 `δ` 加上破坏性噪声 `ξ`：

- **目标特征 δ（Generate target feature）**
  - 优化目标：`δ = argmin_δ E_(x,y)~Di,i∈C [L(F(x+δ;θg), yt)]`，约束 `||δ||∞ ≤ ε`
  - 为了让触发器能针对不同样本自适应（动态触发器），使用一个**生成网络 G_w**（输入 x，输出与 x 同形状的噪声），最后一层用 tanh 激活并乘以 ε 满足约束：`δ = ε · G_w(x)`
- **破坏性噪声 ξ（Craft disruptive noise）**
  - 优化目标：`ξ = argmax_ξ [L(F(x+ξ;θg), y)]`，约束 `||ξ||∞ ≤ σ`
  - 近似解：`ξ = σ · sign(∇_x L(F(x;θg), y))`（类似 FGSM，但目的是使模型更依赖 δ 而不是攻击）

### 2.3 生成网络训练目标
```
min_w E_(x,y)~Di,i∈C [ L( F(x + ε·G_w(x) + σ·sign(∇_x L(F(x;θg),y)) ; θg), yt ) ]
```

### 2.4 算法流程（Algorithm 1: PFL process with Bad-PFL）
```
Server Executes:
  初始化全局模型参数 θg
  while 未收敛 do:
      广播 θg 给被选中客户端 (ClientUpdate)
      聚合上传的参数，形成新的全局模型
  end

ClientUpdate:
  if 该客户端被攻陷 (compromised):
      从攻击者处下载生成网络 G_w，按公式(7)训练
      将训练好的 G_w 返回给攻击者
      基于公式(4)（含后门项，α为投毒率）训练本地模型 F(·;θg)
  else:
      基于公式(4)训练本地模型，但令 α=0（无后门项）
  训练个性化模型（使用预定义的PFL方法）
  将新的 θg 返回服务器
```

### 2.5 生成网络架构（Table 5）
Encoder-Decoder 结构：
- Encoder: 4层 `Conv2d + BatchNorm2d + ReLU`，每层通道数翻倍，kernel size=4, stride=2, padding=1
- Decoder: 4层 `ConvTranspose2d + BatchNorm2d + ReLU`（最后一层用 `Tanh` 激活）

---

## 3. 实验设置

### 3.1 数据集与模型
- 数据集：SVHN, CIFAR-10, CIFAR-100
- 默认模型：ResNet10
- 附加实验模型：ResNet18, ResNet34, MobileNetV2, DenseNet, ViT（预训练于ImageNet，用于评估架构泛化性）

### 3.2 FL 设置
| 参数 | 取值 |
|---|---|
| 客户端总数 | 100 |
| 训练轮数 | 1000 rounds |
| 恶意客户端数 | 10 |
| 每轮参与比例 | 10% 客户端随机被选中 |
| Non-IID 划分 | Dirichlet 分布，因子 = 0.5（默认） |
| 本地优化器 | SGD |
| 本地学习率 | 0.1 |
| 本地 batch size | 32 |
| 本地训练步数 | 15 steps（约1个epoch） |

### 3.3 评测的 PFL 方法（7种）
FedAvg, SCAFFOLD, FedProx, Ditto, FedBN, FedRep, FedPAC

### 3.4 对比的基线后门攻击（6种）
DBA, FCBA, ModRep (Model-Replacement), PGD-Bkd, Neurotoxin, LF-Attack

### 3.5 对比的防御方法
ClipAvg, Multi-Krum, Median, Sign, NAD, I-BAU, Fine-tuning (FT)

### 3.6 评价指标
- **Acc**：干净样本上的平均准确率（%）
- **ASR**（Attack Success Rate）：触发样本上的攻击成功率（%），在客户端的个性化模型 + 测试集上评估

### 3.7 Bad-PFL 专属超参数
| 超参数 | 取值 |
|---|---|
| 投毒率 α（所有攻击统一） | 0.2 |
| ε（δ的约束范围） | 4/255 |
| σ（ξ的约束范围） | 4/255 |
| 生成网络优化器 | Adam |
| 生成网络学习率 | 0.01 |
| 生成网络训练步数 | 30 steps |
| 目标标签 yt | 随机生成 |

### 3.8 基线攻击超参数（Appendix A）
- Ditto / FedProx 正则化强度 R = 0.1
- DBA：触发器尺寸 1×3，从左上角开始排列，间隔1像素，每行5个触发器
- FCBA（DBA改进版）：超参数 m = 4
- ModRep：参数差异放大因子 = 10
- PGD-Bkd：本地-全局参数差异范数约束 = 1
- Neurotoxin：更新幅度最小的底部 10% 参数
- LF-Attack：τ = 0.95（识别后门关键层），λ = 1

### 3.9 防御方法超参数
- ClipAvg：裁剪阈值 t = 1
- Multi-Krum：f = 1（预期恶意客户端数），选取 Krum 距离最高的5个客户端聚合
- Median：逐元素取中位数
- Sign：聚合后取符号，乘以 0.01 更新全局模型

---

## 4. 主要实验结果

### 4.1 CIFAR-10 上各PFL方法的攻击性能（Table 1）
（Acc / ASR，%）

| Attack | FedAvg | SCAFFOLD | FedProx | Ditto | FedBN | FedRep | FedPAC |
|---|---|---|---|---|---|---|---|
| ModRep | 68.00/63.81 | 79.04/54.41 | 76.57/37.58 | 77.42/70.04 | 81.91/27.52 | 79.99/23.38 | 82.49/38.21 |
| Neurotoxin | 79.75/80.53 | 80.09/79.48 | 76.85/71.30 | 78.76/69.00 | 81.11/59.48 | 79.83/28.41 | 81.33/82.10 |
| PGD-Bkd | 79.27/78.61 | 78.95/94.19 | 77.04/74.54 | 78.72/72.23 | 81.21/54.81 | 79.77/20.25 | 81.54/63.45 |
| DBA | 78.97/92.41 | 79.11/94.85 | 77.47/83.59 | 79.70/76.45 | 80.36/31.33 | 79.80/15.54 | 82.46/18.23 |
| FCBA | 78.92/97.33 | 77.82/99.94 | 77.24/88.89 | 78.11/79.20 | 80.54/37.88 | 81.00/16.91 | 83.06/19.91 |
| LF-Attack | 79.85/95.90 | 78.86/95.98 | 76.82/85.46 | 78.20/78.94 | 81.09/44.55 | 80.90/12.82 | 83.25/15.82 |
| **Bad-PFL** | **79.28/99.88** | **78.95/99.80** | **77.36/99.68** | **78.88/94.12** | **80.72/82.22** | **80.29/97.95** | **82.67/99.10** |

**结论**：Bad-PFL 在所有 PFL 方法上的 ASR 都 ≥ 80%，远高于其他基线（尤其在 FedBN/FedRep/FedPAC 这类 partial model-sharing 方法上优势明显）。

### 4.2 对抗SOTA防御（Table 2，CIFAR-10, FedBN/FedRep）
| Attack | PFL | ClipAvg (Acc/ASR) | Multi-Krum (Acc/ASR) | Median (Acc/ASR) | Sign (Acc/ASR) |
|---|---|---|---|---|---|
| Bad-PFL | FedBN | 82.55/82.66 | 66.93/**80.28** | 74.44/50.52 | 31.42/24.13 |
| Bad-PFL | FedRep | 81.59/**97.28** | 70.41/**96.15** | 70.23/77.21 | 34.49/20.32 |

**结论**：Median 防御效果最好，但仍无法将 Bad-PFL 的 ASR 压到 80% 以下（FedBN: 50.52%, FedRep: 77.21%）。

### 4.3 后门持久性（Table 3，CIFAR-10）
FT-15/30/45 = 微调15/30/45步：
- Bad-PFL 在 FedBN 下，微调后 ASR 仍保持 80%+（Before: 82.22 → FT-45: 80.12）
- Bad-PFL 在 FedRep 下，微调后 ASR 仍保持 97%+（Before: 97.95 → FT-45: 97.01）
- 对比 Neurotoxin/LF-Attack 微调后 ASR 大幅下降（如 FedBN Neurotoxin: 59.48 → 18.35）

### 4.4 消融实验（Table 4：δ 和 ξ 的贡献，FedBN/FedRep）
| Component | FedBN Acc/ASR | FedRep Acc/ASR |
|---|---|---|
| w/o δ | 80.78/13.74 | 80.17/12.82 |
| w/o ξ | 80.56/68.56 | 80.20/79.32 |
| Both（完整） | 80.72/82.22 | 80.29/97.95 |

### 4.5 敏感性分析（ε, σ的影响，Table 18）
- Bad-PFL 对 ε（δ幅度）更敏感（ε=0时ASR仅13.74%/12.82%，ε=4时达到82.22%/97.95%）
- 对 σ（ξ幅度）相对不敏感（σ=0时ASR仍有68.56%/79.32%）

### 4.6 其他关键结果
- **单恶意客户端场景**（Table 3上下文）：仅1个恶意客户端时，Bad-PFL 仍达到 94.01% ASR，Neurotoxin/LF-Attack 则很低
- **不同模型架构**（Table 14）：ResNet10/18/34, MobileNetV2, DenseNet 上 Bad-PFL 均保持高ASR（75%-99%区间）
- **ViT架构**（Table 15，FedRep）：Bad-PFL ASR = 98.94%，Neurotoxin = 50.64%，LF-Attack = 20.58%
- **SVHN & CIFAR-100**（Table 16,17）：Bad-PFL 同样全面超越基线
- **多目标攻击**（Table 20）：训练多个生成器（每类别一个），性能略有下降但ASR基本不受影响
- **攻击成本**（Table 23，CIFAR-10, RTX4090）：Bad-PFL 单次本地训练耗时 0.613s~1.206s（视PFL方法而定），远低于 Perdoor（3.1s~5.7s）与 PFedBA（1.44s~1.82s）
- **隐蔽性评估**（Table 22，Neural Cleanse + STRIP）：Bad-PFL anomaly index = 2.2（越低越隐蔽，对比 Neurotoxin 5.8, PFedBA 4.9），entropy = 0.77（越高越隐蔽，对比 Neurotoxin 0.13, PFedBA 0.25）

---

## 5. 面向 Claude Code 复现的关键要点摘要

1. **威胁模型**：白盒攻击，攻击者完全控制被攵陷客户端的本地训练过程，但无法控制服务器聚合规则或良性客户端。
2. **触发器生成**是核心创新：需要实现一个 encoder-decoder 生成网络 G_w（见架构表），并与全局模型交替优化。
3. **两阶段优化**：
   - 阶段1：优化生成网络参数 w（30步，Adam，lr=0.01），目标见公式(7)
   - 阶段2：用生成的触发器训练本地模型（含正常任务损失+后门任务损失，投毒率α=0.2）
4. 需要实现的 PFL baseline：至少需要 FedBN 和 FedRep（论文中重点讨论的两个 partial-sharing 方法）
5. 核心比较对象 Neurotoxin 和 LF-Attack 是最强基线，复现时应作为对照
6. 关键超参数速查：`ε=σ=4/255`，本地SGD lr=0.1，batch=32，本地步数=15，Dirichlet α=0.5，100 clients/10 malicious/1000 rounds
