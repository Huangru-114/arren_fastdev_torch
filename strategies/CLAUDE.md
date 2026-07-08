# strategies/ —— Layer 2，唯一写业务逻辑的地方

- `base.py`：框架契约。`ExperimentConfig`/`RunResult`、三个扩展点
  (`PFLMethod`/`AttackHook`/`DefenseHook`)、以及算法无关的 `Client`/`Server`/`FLEngine`。
  执行顺序在此固定：客户端阶段跑 attack hook，服务端聚合阶段跑 defense hook；
  pFL 部分共享通过在 upload/distribute 两侧裁剪私有参数实现。
- `registry.py`：`@register_pfl` / `@register_attack` / `@register_defense` + 查找函数。
- `attacks/atk1.py`：Bad-PFL（ICLR2025）攻击，δ=ε·G(x) 生成器 + ξ 破坏噪声 + 投毒/ASR 变换。
- `pfl/fedbn.py`、`pfl/fedrep.py`：两个部分共享 pFL 方法（FedBN 私有 BN；FedRep 私有分类头 + 两阶段本地训练）。
- `all.py`：import 即注册所有内置策略，供上层按名字派发。

规则：只能 `from pfl_lib.interface import ...` 调基础设施，**禁止**深层路径与相对导入；
每个策略文件顶部 docstring 注明对应 `papers/` 来源。新增策略后跑 `python experiments/smoke_test.py`。
