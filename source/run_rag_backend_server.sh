#!/bin/sh

set -eu

if [ $# -lt 4 ]; then
  echo "Usage: $0 /path/to/model /path/to/data_dir /path/to/embedder /path/to/reranker [extra args]"
  exit 1
fi

MODEL_PATH="/workspace/data/demo_13_02/RRNCB/models/meno-lite-0.1"
DATA_DIR="/workspace/data/demo_13_02/RRNCB/data"
EMBEDDER_DIR="/workspace/data/demo_13_02/RRNCB/models/FRIDA"
RERANKER_DIR="/workspace/data/demo_13_02/RRNCB/models/bge-reranker-v2-m3"

SCRIPT_DIR="$(cd "$(dirname "/workspace/data/demo_13_02/RRNCB")" && pwd)"

python3 -u "${SCRIPT_DIR}/rag_backend_server.py" \
  --model "${MODEL_PATH}" \
  --data-dir "${DATA_DIR}" \
  --embedder "${EMBEDDER_DIR}" \
  --reranker "${RERANKER_DIR}" \
  --gpu "${GPU_MEM_PART:-0.85}" \
  "$@"
