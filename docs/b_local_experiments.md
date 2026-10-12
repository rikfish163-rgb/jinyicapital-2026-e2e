# B 的本地分区训练与实验

`ptop.local_training` 把下载的月度 Feather 接到 A 的 raw25、冻结处理器和 `WindowBatch`。月文件逐个 Arrow batch 读取，请求的日期和股票被写入私人每日 raw cache；训练每次仅持有一天的原始、规范化和窗口索引。重叠窗口在取 batch 时展开，学习后的 encoder token 不进入缓存。

实际训练入口为 `prepare_local_fold`。它必须得到官方 expected 采样键、`instrument_map`、`LabelSpec`、显式分钟网格和 `TimeFold`，以及 A 合同格式的 targets，或官方 labels 加显式 support。没有这些信息会在读取行情前报出缺项。四份原始行情不能提供股票正式代码、官方有效采样集合或未经确认的目标定义。

```python
from ptop.local_training import LocalBarStore, prepare_local_fold

store = LocalBarStore(bar1m_dir, cache_dir=private_cache_dir)
prepared = prepare_local_fold(
    store, official_expected, official_targets,
    label_spec=label_spec, trading_grid=trading_grid, fold=fold,
    instrument_map=instrument_map, window_length=120,
    scale_review="实际尺度检查记录；如未人工检查需如实注明",
)
# 与 Trainer 的结构接口一致，窗口数组仍由 A 构造。
train_batches = prepared.train_loader(epoch=0)
validation_batches = prepared.validation_loader()
```

本地 fold 按午夜边界切分，保留完整 expected 集合。目标支持区间到达下一划分边界时，target 标为无效，并记录 `target_reaches_split_boundary`；不会因标签缺失删除官方采样键。行情缺失和实际模型字段全空由 A 的 predictable 判定分流。

N1 只使用训练端点可见的唯一原始行。每一天先通过 A 的 `fit_row_union` 取得引用行，再用 float64 的 pooled mean/M2 汇合各列 population mean/std。零方差列采用锁定 Qlib `ZScoreNorm` 的 mean=0、std=1、ignore=True 语义。状态记录 field order、valid_counts、排序唯一行指纹、实现版本、真实尺度检查说明，再用 A 的 `FrozenPreprocessor.from_state_dict` 恢复。这个路径是分区等价实现，不能称为调用了 A 的整表 `fit`。测试将它与实际 A/Qlib 小样本拟合比较；验证和推理仅复用冻结状态。

模型对比先用最长窗口拟合一份共同 N1，再用 `prepared.with_window(60)` 和 `prepared.with_window(120)`。所有候选保持相同采样键、目标有效性和目标值。调度器核验每次验证产物的完整 sample_id 集合、实际窗口 predictable 和有效预测键；不同覆盖不会被悄悄取交集。

`ReadPolicy` 默认拒绝未知负值。下载文件中的盘口委托笔数是有符号 int16；显式 `negative_order_policy="mask_known"` 时，仅六个 `*_num_orders{1,2,3}` 的负值会改为缺失并计入读取审计。不会取绝对值、转换 uint16、加 65536 或推测原值。价格、金额和成交量等其他负值继续由 A 拒绝。正数笔数保持真实值，缺失 mask 继续进入 encoder。

可以在固定训练依赖环境执行真实行情工程诊断：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/data4/hts/jyc2026_training_env/bin/python \
  examples/b_real_data_smoke.py --config configs/b_diagnostic_cpu.json \
  --output-dir /home/data4/hts/jyc2026/b_runs/diagnostic-v1 --mode matrix --threads 2
```

配置使用 2020-01-02、01-03 训练，01-06 验证，原始 ID 为 11、16、19、30。`10:00/11:00/14:00` 是显式工程端点，只能证明它们在观察网格中，不能当作官方合法采样点。`bar_end_is_available=true` 是这次诊断的可见性假设，元数据明确保留 `production_visibility_verified=False`。

诊断目标是 sample_id 的 SHA256 映射到固定数值，**不是未来收益，不是官方标签，诊断 loss/IC 不代表金融预测效果**。默认矩阵为同一个 A ALSTM encoder 的 B-GRU、B-Transformer、B-TCN、B-SelectiveSSM，L=60/120、seeds=7/17/29，各 2 epochs、独立 600 秒预算。可用 `--models`、`--lookbacks`、`--seeds` 和 `--epochs` 选择有明确独立预算的子矩阵。

产物位于私人 output_dir：`prepared_manifest.json`、`normalization.json`、`split_audit.parquet`、真实读取/前向的 `benchmark.json`，以及 matrix 下的训练日志、checkpoint、完整验证结果、`comparison.csv`、`seed_summary.csv` 和 `report.md`。报告明确区分已完成、失败和预算超限；超限不替换成更简单模型，只有上一完整 epoch 的 checkpoint 保证可恢复。`--resume-completed` 会验证已完成实验的协议和验证文件哈希后跳过。

benchmark 分别记录 loader 等待、张量传输、前向耗时、样本吞吐和内存。CUDA 测量显式同步并重置分配峰值；CPU RSS 是整个进程的累计峰值，不冒称为某一个模型独立占用。没有官方标签时不计算或宣称平台 BARRA、压力场景、换手百分位或五项总分。

正式入口模板为 `configs/b_official_template.json`。null 项需要填入真正的官方文件；它们是可操作的缺项而非可训练的假数据。正式标签准备好后，保持相同时间折和目标合同运行矩阵，选择开发验证候选，再由模型包与提交入口冻结交接。

统一 `.py` CLI 也可选择 `data_factory="ptop.local_training:local_training_factory"`，data 中填写上述官方文件路径及私人 cache_dir；正式 factory 的窗口字段取真实 model.encoder.fields。工程诊断可运行 `python -m ptop.train --config configs/b_train_cli_diagnostic.json --output /私人目录/cli-smoke --device cpu --threads 2`。此配置显式选择 `diagnostic_training_factory`，输出可恢复训练 checkpoint，但不导出包含假股票代码的正式模型包。

当前验证记录将由真实运行追加；只有确实执行并检查的结果才写入实验结论。
