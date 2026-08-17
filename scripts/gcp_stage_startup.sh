#!/usr/bin/env bash
set -euo pipefail

exec > >(tee -a /var/log/gcp-stage-startup.log) 2>&1

meta() {
  curl -fsH "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

CFG_NAME="$(meta CFG_NAME)"
RUN_NAME="$(meta RUN_NAME)"
PROJECT_ID="$(meta PROJECT_ID)"
BUCKET_URI="$(meta BUCKET_URI)"
SOURCE_OBJECT="$(meta SOURCE_OBJECT)"
BASE_RUN_OBJECT="$(meta BASE_RUN_OBJECT)"
MAX_MINUTES="$(meta MAX_MINUTES)"
STAGE_NAME="$(meta STAGE_NAME)"
EVAL_DEVICE="$(meta EVAL_DEVICE || echo auto)"

DEST_BASE="$BUCKET_URI/$STAGE_NAME/$RUN_NAME"

TRAIN_LOG="train_${RUN_NAME}.log"
EVAL_LOG="eval_${RUN_NAME}.log"
EVAL_MD="eval_${RUN_NAME}.md"
LATEST_RUN=""

upload_artifacts() {
  set +e
  if [ -f "$TRAIN_LOG" ]; then
    gcloud storage cp "$TRAIN_LOG" "$DEST_BASE/logs/"
  fi
  if [ -f "$EVAL_LOG" ]; then
    gcloud storage cp "$EVAL_LOG" "$DEST_BASE/logs/"
  fi
  if [ -f "$EVAL_MD" ]; then
    gcloud storage cp "$EVAL_MD" "$DEST_BASE/reports/"
  fi
  if [ -n "$LATEST_RUN" ] && [ -d "$LATEST_RUN" ]; then
    gcloud storage rsync -r "$LATEST_RUN" "$DEST_BASE/run/"
  fi
  if [ -f /var/log/gcp-stage-startup.log ]; then
    gcloud storage cp /var/log/gcp-stage-startup.log "$DEST_BASE/logs/"
  fi
}

trap upload_artifacts EXIT

WORKDIR="/opt/prometheus"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

apt-get update
apt-get install -y python3-venv python3-pip unzip jq

# Pull exact source snapshot created by launcher.
gcloud storage cp "$SOURCE_OBJECT" source.zip
unzip -q source.zip -d repo
cd repo

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

mkdir -p outputs

# Pull base run artifacts required by evaluation and latent training.
gcloud storage cp -r "$BASE_RUN_OBJECT" outputs/
BASE_RUN_DIR="outputs/$(basename "$BASE_RUN_OBJECT")"

python - "$CFG_NAME" "$RUN_NAME" "$BASE_RUN_DIR/checkpoint.pt" <<'PY'
import sys
import yaml
from pathlib import Path

cfg_name = sys.argv[1]
run_name = sys.argv[2]
base_checkpoint = sys.argv[3]

cfg_path = Path("configs") / cfg_name
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cfg["model"]["base_checkpoint"] = base_checkpoint
cfg.setdefault("experiment", {})["run_name"] = run_name
cfg["experiment"]["output_dir"] = "outputs"

active = Path("configs") / f"_active_{cfg_name}"
with active.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(active)
PY

ACTIVE_CFG="configs/_active_${CFG_NAME}"

ok=0
for attempt in 1 2; do
  set +e
  timeout "${MAX_MINUTES}m" ./.venv/bin/python -u -m prometheus.cli train --config "$ACTIVE_CFG" > "$TRAIN_LOG" 2>&1
  code=$?
  set -e
  if [ "$code" -eq 0 ]; then
    ok=1
    break
  fi
  echo "train attempt ${attempt} failed with exit ${code}"
done

LATEST_RUN=""
if [ "$ok" -eq 1 ]; then
  LATEST_RUN=$(ls -td outputs/${RUN_NAME}-* 2>/dev/null | head -n1 || true)
  if [ -n "$LATEST_RUN" ]; then
    set +e
    ./.venv/bin/python -u -m prometheus.cli evaluate-reasoning \
      --base-run "$BASE_RUN_DIR" \
      --latent-run "$LATEST_RUN" \
      --num-problems 300 \
      --device "$EVAL_DEVICE" \
      --output "$EVAL_MD" > "$EVAL_LOG" 2>&1
    set -e
  fi
fi

shutdown -h now
