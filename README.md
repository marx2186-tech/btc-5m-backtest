# BTCUSDT 5分钟自动回测

本项目由 GitHub Actions 自动执行：

1. 下载 Binance 官方 USDⓈ-M 永续合约 2026-05、2026-06 的 BTCUSDT 5分钟月度K线。
2. 校验官方 SHA-256，并审计 17,568 根K线：零缺失、零重复、OHLC合法。
3. 搜索多种趋势、突破、回踩、均值回归与多因子评分组合。
4. 验收目标：每月20–30笔、每月胜率≥70%、每月净平均盈利/平均亏损≥1.5。
5. 计入单边0.05%手续费与2个价格跳动滑点。
6. 将报告、逐笔交易和 Pine Script v6 写入 `results/`。

上传文件后，打开仓库的 **Actions** 页面查看运行状态。运行完成后直接查看：

- `results/report.md`
- `results/trades.csv`
- `results/best_config.json`
- `results/BTC_5m_optimized_strategy.pine`

若没有任何策略同时达到全部硬性条件，报告会明确标记未达标，并给出最接近的候选，不会篡改统计。
