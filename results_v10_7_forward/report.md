# BTCUSDT V10.7 追加式真实前向观察报告

- 参数冻结时间线起点：**2026-07-29T00:00:00Z**。
- 本次数据截止：**2026-07-27T23:55:00+00:00**，仅包含完整UTC日。
- 空头是唯一主观察通道；多头与原多空组合只记录影子结果，不参与参数选择。
- 不搜索参数、不替换指标、不自动宣布实盘合格。

## 当前状态

- 状态：**WAITING_FOR_TRUE_FORWARD_START**。
- 已积累完整前向日：**0** 天。
- 空头已完成交易：**0 / 20** 笔。
- 样本进度：**0.0%**。
- 实盘资格：**否**。

## 三条冻结观察通道

| 通道 | 角色 | 完成交易 | 胜率 | 实际盈亏比 | PF | 净R | 最大回撤 | 删除最佳10%后 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V10.7空头主观察·冻结ADX衰减≥-4 RR2.5 | PRIMARY_FORWARD_OBSERVATION | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| V10.7多头影子观察·冻结ADX≤45 RR2.5 | SHADOW_ONLY | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| V10.7原多空组合影子观察·共享单一仓位 | SHADOW_ONLY | 0 | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## 空头主观察门槛

| 检查 | 结果 |
|---|---|
| minimum_closed_trades | 未通过/样本不足 |
| minimum_win_rate | 未通过/样本不足 |
| minimum_avg_win_loss_ratio | 未通过/样本不足 |
| minimum_profit_factor | 未通过/样本不足 |
| positive_expectancy | 未通过/样本不足 |
| maximum_drawdown_R | 通过 |
| best_10pct_removed_still_profitable | 未通过/样本不足 |
| minimum_cost_1_5x_profit_factor | 未通过/样本不足 |

## 当前未完成持仓快照

当前没有尚未达到止损、止盈或完整时间退出条件的观察持仓。

## 追加式账本审计

- 完成交易账本新增：**0** 条。
- 信号账本新增：**0** 条。
- 日快照新增：**0** 条。
- 既有记录被修改或删除：**0**。

## 纪律

在空头完成交易少于20笔前，只积累证据，不评价参数优劣。达到20笔后也只执行预先写死的门槛检查；V10.7不会根据前向结果自动调参。

重点文件：`forward_trade_ledger.csv`、`forward_signal_ledger.csv`、`daily_observation_snapshots.csv`、`open_trade_snapshot.csv`、`forward_summary.csv`、`cost_stress_summary.csv`、`primary_forward_audit.json`、`append_only_audit.json`、`strategy_lock.json`。
