from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
status = json.loads((ROOT / "results/status.json").read_text(encoding="utf-8"))
report = (ROOT / "results/report.md").read_text(encoding="utf-8")
icon = "✅" if status.get("qualified") else "⚠️"
print(f"{icon} 自动回测已完成\n\n{report}\n\n结果文件已写入仓库 results/，包括逐笔交易、候选排行、参数与 Pine v6。")
