#!/usr/bin/env bash
# ============================================================
# FraudSentinel — one-time dependency installer
# AMD ROCm hackathon image (vLLM 0.11.0rc2 pre-installed)
#
# Run once from the repo root:
#   bash scripts/install.sh
# ============================================================
set -e

export WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")/.." && pwd)}"

echo "==> [1/4] Fixing blinker conflict..."
pip install --ignore-installed blinker

echo ""
echo "==> [2/4] Pinning starlette/protobuf/numpy for vLLM coexistence..."
pip install "starlette>=0.40.0,<0.49.0" "protobuf<7.0.0" "numpy<2.3"

echo ""
echo "==> [3/4] Installing FraudSentinel dependencies..."
pip install \
  "fastapi>=0.111.0" \
  "uvicorn[standard]>=0.29.0" \
  "httpx>=0.27.0" \
  "lightgbm>=4.3.0" \
  "scikit-learn>=1.4.0" \
  "joblib>=1.3.0" \
  "pandas>=2.1.0" \
  "huggingface_hub>=0.23.0"

echo ""
echo "==> [4/4] Downloading Tier-1 scorer model files..."

mkdir -p "$WORKSPACE/models/card_fraud" "$WORKSPACE/models/aml"

python3 - <<'PYEOF'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

workspace = Path(os.environ["WORKSPACE"])

# HF repo layout maps directly to models/ subdirs
files = {
    "Card Fraud Production Scorer/cc_lgbm_model.txt":      workspace / "models/card_fraud/cc_lgbm_model.txt",
    "Card Fraud Production Scorer/cc_lgbm_preproc.joblib": workspace / "models/card_fraud/cc_lgbm_preproc.joblib",
    "Card Fraud Production Scorer/cc_lgbm_metrics.json":   workspace / "models/card_fraud/cc_lgbm_metrics.json",
    "AML Tabular Baseline/aml_lgbm_model.txt":             workspace / "models/aml/aml_lgbm_model.txt",
    "AML Tabular Baseline/aml_lgbm_preproc.joblib":        workspace / "models/aml/aml_lgbm_preproc.joblib",
    "AML Tabular Baseline/aml_lgbm_metrics.json":          workspace / "models/aml/aml_lgbm_metrics.json",
}

for hf_path, local_path in files.items():
    if local_path.exists():
        print(f"  ✓ already present: {local_path.name}")
        continue
    print(f"  ↓ downloading {local_path.name} ...")
    # Download to a temp location then move to avoid partial files
    import tempfile, shutil
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = hf_hub_download(
            repo_id="naazimsnh02/fraudsentinel-tier1-scorers",
            filename=hf_path,
            local_dir=tmp,
            local_dir_use_symlinks=False,
        )
        shutil.copy2(downloaded, local_path)
    print(f"  ✓ {local_path.name}")
PYEOF

echo ""
echo "==> Done."
echo ""
echo "    Start vLLM (Terminal 1):"
echo "      bash scripts/start.sh vllm"
echo ""
echo "    Start app (Terminal 2):"
echo "      bash scripts/start.sh app"
echo ""
echo "    Open browser:"
echo "      https://notebooks.amd.com/<your-pod>/proxy/7860/"
