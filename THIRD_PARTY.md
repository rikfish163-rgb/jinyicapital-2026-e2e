# 第三方来源

Qlib 固定提交为 [54355232463878d2eebb91fe0ee5fa7fa1f5976c](https://github.com/microsoft/qlib/tree/54355232463878d2eebb91fe0ee5fa7fa1f5976c)，使用 MIT 许可。通过依赖直接调用其模型、处理器、DataHandlerLP、StaticDataLoader 和 TSDataSampler；源码及许可随对应依赖安装。

DeepLOB 分支依据 [DeepLOB 论文](https://arxiv.org/html/1808.03668)和三档输入配置，独立组织 PyTorch 卷积、Inception、LSTM 与回归头。Patch 分支参考 [PatchTST 论文](https://arxiv.org/html/2211.14730)的线性分块形式，后接 GRU 与注意力。这些分支的本赛道效果需实际训练确定。

B 的 Transformer 与 GRU 使用 PyTorch 模块；TCN 和 SelectiveSSM 为独立组织的研究实现，参考 [TCN](https://arxiv.org/abs/1803.01271) 与 [Mamba](https://arxiv.org/abs/2312.00752) 的结构思想，没有携带 Mamba CUDA 扩展或外部预训练权重。论文、官方实现与本实现的具体区别见 [时序研究](docs/b_temporal_research.md)。模型包记录完整环境版本，并携带五个核心依赖的 LICENSE/NOTICE；Qlib 许可副本来自上述固定提交并以 SHA-256 核对。
