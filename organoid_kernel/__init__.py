"""organoid_kernel —— 自包含的多数据集具身数据验证内核。

按 introduction.md 的架构实现:无损 Adapter → Canonical Evidence → 能力清单 →
Profile/Policy → Planner → 逐 claim 收据 → Claim Ledger → 分级裁决。
不依赖本包之外的任何 organoid 代码;机器人资产与 golden 收据内置。
"""
__version__ = "0.1.0"
SCHEMA_PREFIX = "organoid-kernel"
