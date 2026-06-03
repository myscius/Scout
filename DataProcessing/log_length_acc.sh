
#!/usr/bin/env bash
set -euo pipefail

# Script to run the log length accuracy extraction command
# Usage: ./log_length_acc.sh
#
# 示例：3 种模式执行示例
# 1) run (先 extract 再 stats)：
#    PYTHON_CMD=python3 ./log_length_acc.sh run \
#      --logs LongVideoBench_gemma4-31b_frame20.log \
#      --output LongVideoBench_gemma4-31b_frame20_extract.json \
#      --bins 0,60,180,300,600,1200 \
#      --stats-output LongVideoBench_gemma4-31b_frame20_stats.json
#
# 2) extract（仅提取结构化结果）：
#    PYTHON_CMD=python3 ./log_length_acc.sh extract \
#      --logs LongVideoBench_gemma4-31b_frame20.log \
#      --output LongVideoBench_gemma4-31b_frame20_extract.json
#
# 3) stats（从 extract 产物或直接从 logs 统计）：
#    从 extract 产物：
#    PYTHON_CMD=python3 ./log_length_acc.sh stats \
#      --input LongVideoBench_gemma4-31b_frame20_extract.json \
#      --bins 0,60,180,300,600,1200 \
#      --stats-output LongVideoBench_gemma4-31b_frame20_stats.json
#
#    直接从日志：
#    PYTHON_CMD=python3 ./log_length_acc.sh stats \
#      --logs LongVideoBench_gemma4-31b_frame20.log \
#      --bins 0,60,180,300,600,1200 \
#      --stats-output LongVideoBench_gemma4-31b_frame20_stats.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD=${PYTHON_CMD:-python}

"$PYTHON_CMD" "$SCRIPT_DIR/log_length_accuracy.py" run \
	--logs LongVideoBench_gemma4-31b_frame20.log \
	--output LongVideoBench_gemma4-31b_frame20_extract.json \
	--bins 0,60,180,300,600,1200 \
	--stats-output LongVideoBench_gemma4-31b_frame20_stats.json

echo "Done. Outputs: LongVideoBench_gemma4-31b_frame20_extract.json, LongVideoBench_gemma4-31b_frame20_stats.json"
