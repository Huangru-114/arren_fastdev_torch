# pfl-lib/ —— Layer 1 基础设施（只读，门面调用）

- 唯一对外入口是 `pfl_lib/interface.py`：`build_model`、`build_dataset`/`DatasetBundle`、
  `partition_data`、`get_aggregator`。上层只能 import 这些名字，**禁止**深层导入 `pfl_lib._*`。
- `_models.py`：CIFAR ResNet10/18/34；命名刻意稳定（BN=`bn1/bn2/shortcut.1`，头=`linear`），
  好让部分共享 pFL 按 key 选私有参数。
- `_partition.py`：`dirichlet`（强度 `alpha`）与 `pathological`（强度 `class_per_client`）两种
  非 IID 划分，逻辑移植自 PFLlib `dataset_utils.separate_data` 的 `dir`/`pat`，改写为返回索引。
- `_datasets.py`：torchvision 加载 cifar10/cifar100/svhn；`synthetic` 供 Tier A 冒烟测试。
- `_aggregators.py`：`avg`/`median`（只聚合各客户端都上传的公共 key）。

未经用户明确许可**不得修改本目录**；需要新功能请在 `strategies/` 里继承/封装。
