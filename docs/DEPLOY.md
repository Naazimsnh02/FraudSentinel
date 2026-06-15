# FraudSentinel — Deployment Guide
## AMD Hackathon Environment (MI300X, ROCm 7.0, vLLM 0.11.0rc2)

---

## Quick start (copy-paste)

**Step 1 — clone & install** (any terminal):
```bash
git clone <repo-url> FraudSentinel
cd FraudSentinel
bash scripts/install.sh
```
This installs Python dependencies and downloads Tier-1 scorer model files from HuggingFace into `models/`.

**Step 2 — Terminal 1: vLLM**
```bash
bash scripts/start.sh vllm
# Wait for: "Application startup complete." (~2–5 min; first run downloads ~30 GB)
```

**Step 3 — Terminal 2: App**
```bash
bash scripts/start.sh app
```

**Step 4 — Open browser:**
```
https://notebooks.amd.com/<your-pod>/proxy/7860/
```
Your pod name is the hostname segment of your Jupyter URL, e.g.:
```
https://notebooks.amd.com/jupyter-hack-team-121-260610051314-8b731bf3/lab
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          this part is your pod name
```

---

## API architecture

```
Browser  ──→  /proxy/7860/              FastAPI serves the SPA dashboard (HTML/JS/CSS)
Browser  ──→  /proxy/7860/api/*         FastAPI REST endpoints
                   ├─ POST /api/score/card     LightGBM card fraud scorer
                   ├─ POST /api/score/aml      LightGBM AML scorer
                   ├─ POST /api/analyze/stream vLLM SSE streaming proxy
                   ├─ POST /api/analyze        vLLM non-streaming (SAR drafts)
                   └─ GET  /api/health         component status

vLLM     ──→  localhost:8000             internal inference engine, not browser-facing
```

The UI is a self-contained single-page application served directly by FastAPI as an HTML
response. All API calls from the browser go to the same origin and port — no CORS
configuration or proxy path mapping needed.

---

## UI tabs

| Tab | Purpose |
|-----|---------|
| **⚡ Score** | Paste any transaction JSON → instant Tier-1 risk score, badge, signals |
| **🧠 Analyze** | Streaming Qwen3-14B analysis — structured JSON, NL explanation, or recommendation |
| **📝 SAR Draft** | Auto-generate FinCEN SAR narrative (non-streaming, editable) |
| **🕸 AML Graph** | D3.js force-directed transaction network with risk coloring |
| **▶ Live Demo** | One-click end-to-end: 3 pre-loaded scenarios (fraud / AML / legit) |

---

## Port reference

| Port | Process | Access |
|------|---------|--------|
| 8000 | vLLM | Internal only (no proxy needed) |
| 7860 | FastAPI (app + UI) | `proxy/7860/` |

---

## Repository layout

```
FraudSentinel/
├── README.md
├── PRD.md
├── .gitignore
├── backend/               ← FastAPI app
│   ├── app.py
│   ├── requirements.txt
│   ├── static/            ← SPA HTML/CSS/JS (Python modules)
│   └── tier1/             ← LightGBM scorer wrappers
├── models/                ← downloaded by install.sh, gitignored
│   ├── card_fraud/        ← cc_lgbm_model.txt, cc_lgbm_preproc.joblib
│   └── aml/               ← aml_lgbm_model.txt, aml_lgbm_preproc.joblib, aml_gnn.pt
├── training/              ← model training scripts
│   ├── card_fraud/        ← cc_lgbm.py
│   └── aml/               ← aml_lgbm.py, train_gnn_aml.py
├── notebooks/             ← Qwen3-14B SFT training notebook
├── scripts/
│   ├── install.sh         ← one-time setup
│   └── start.sh           ← launch vLLM or app
└── docs/
    └── DEPLOY.md          ← this file
```

Override model paths with environment variables if needed:
```bash
export CARD_SCORER_DIR=/custom/path/to/card/models
export AML_LGBM_DIR=/custom/path/to/aml/models
export FRAUDSENTINEL_MODEL=naazimsnh02/fraudsentinel-qwen3-14b-merged
```

---

## Using cached model weights (skip HF download)

`scripts/start.sh vllm` auto-detects the local HuggingFace cache and uses it if present — no extra steps needed. To pass a path explicitly:

```bash
vllm serve ~/.cache/huggingface/hub/models--naazimsnh02--fraudsentinel-qwen3-14b-merged/snapshots/7caf75818541f7fa95eabf5815d27dbd46dc21b3 \
  --dtype bfloat16 --max-model-len 4096 --port 8000
```

---

## Troubleshooting

**vLLM fails with "Invalid repository ID"**
HuggingFace is unreachable. Set `HF_HUB_OFFLINE=1` and pass the local snapshot path directly (see above).

**vLLM shows `_C.abi3.so` import warnings**
Expected on ROCm 7.0, safe to ignore. The server starts normally.

**Health check shows vLLM ✗ in the header**
vLLM is still loading or not started. Watch Terminal 1 for `Application startup complete`.

**Scorer shows ✗ (red dot)**
Model files missing from `models/card_fraud/` or `models/aml/`. Re-run `bash scripts/install.sh`.

**Page loads but Analyze / SAR returns 404**
The app auto-detects the vLLM model ID at startup. If vLLM was started after the app, restart the app.
