# 🛡️ FraudSentinel

### Enterprise Fraud Detection & Financial Crime Intelligence Platform

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Status](https://img.shields.io/badge/status-complete%20%26%20deployed-brightgreen)](#-project-status)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-naazimsnh02-orange)](https://huggingface.co/naazimsnh02)
[![Base Model](https://img.shields.io/badge/base%20model-Qwen3--14B-purple)](https://huggingface.co/Qwen/Qwen3-14B)
[![Dataset](https://img.shields.io/badge/dataset-11.8k%20conversations-green)](https://huggingface.co/datasets/naazimsnh02/fraud-financial-crime-qwen3-sft-v2)
[![Framework](https://img.shields.io/badge/fine--tuning-Unsloth%20%2B%20TRL-red)](https://github.com/unslothai/unsloth)
[![Hardware](https://img.shields.io/badge/hardware-AMD%20MI300X-EE0000)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
[![Demo](https://img.shields.io/badge/▶%20YouTube-Demo-red)](https://youtu.be/YvvmbmW6nUs)

---

## 📋 Overview

**FraudSentinel** is a two-tier fraud detection and financial crime intelligence platform that combines **ultra-fast statistical models** for real-time triage with a **fine-tuned LLM (Qwen3-14B)** for explainable, investigator-facing risk analysis — across both **card-not-present (CNP) fraud** and **anti-money laundering (AML)**.

> Unlike pure-LLM or pure-statistical approaches, FraudSentinel uses the right tool for the right job: sub-10ms triage at scale for 100% of transactions, and deep reasoning + explanation only for the small fraction that get flagged.

### Key Capabilities

- ⚡ **Real-time triage** — LightGBM + GNN models score transactions in <10ms with >50K TPS throughput
- 🧠 **Explainable AI** — Fine-tuned Qwen3-14B generates structured risk JSON, natural-language explanations, and typology classification
- 🕸️ **Graph-aware AML detection** — GINE-based GNN catches multi-hop laundering patterns (fan-out, cycles, structuring) invisible to tabular models
- 📝 **SAR drafting** — Auto-generates FinCEN-compliant Suspicious Activity Report drafts for human review
- 👤 **Human-in-the-loop** — 6-level investigator action taxonomy
- 🔍 **Deep Analysis mode** — Optional Chain-of-Thought reasoning for complex multi-account cases

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  INCOMING TRANSACTION STREAM                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1 — REAL-TIME TRIAGE  (<10ms, 100% coverage)              │
│   • Card Scorer (LightGBM)    — PR-AUC 0.967, ROC-AUC 0.999    │
│   • AML Pre-Filter (LightGBM) — ROC-AUC 0.822, PR-AUC 0.023    │
│   • AML GNN (GINE)            — graph topology: fan-out/fan-in/cycles │
└─────────────────────────────────────────────────────────────────┘
                              │  [flagged transactions only]
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 2 — INTELLIGENT EXPLANATION  (async, fine-tuned Qwen3)    │
│   • Structured risk JSON (score, typology, signals)             │
│   • Investigator-facing natural language explanation            │
│   • 6-level recommended action                                  │
│   • SAR draft generation                                        │
│   • Deep Analysis (CoT) for complex cases                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  INVESTIGATOR DASHBOARD  (browser SPA)                          │
│   Score · Analyze · SAR Draft · AML Graph · Live Demo           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Components

| Component | Type | Status | Metrics |
|---|---|---|---|
| **Card Fraud Scorer** | LightGBM | ✅ Complete | PR-AUC 0.967, ROC-AUC 0.999 |
| **AML Pre-Filter** | LightGBM | ✅ Complete | ROC-AUC 0.822, PR-AUC 0.023 |
| **AML GNN** | GINEConv GNN | ✅ Complete | ROC-AUC 0.584, PR-AUC 0.0036 ¹ |
| **Tier-2 LLM (Qwen3-14B + LoRA)** | Fine-tuned LLM | ✅ Complete | Train loss 0.247, 70.5 min on MI300X |
| **REST API** | FastAPI | ✅ Complete | Score, Analyze, SAR, Health endpoints |
| **Investigator Dashboard** | HTML/JS SPA | ✅ Complete | 5-tab UI, streaming, D3.js AML graph |

---

## 🖥️ Investigator Dashboard

A single-page application served directly by FastAPI — no separate frontend build step required.

| Tab | Purpose |
|-----|---------|
| **⚡ Score** | Paste transaction JSON → instant Tier-1 risk score, badge, and feature signals |
| **🧠 Analyze** | Streaming Qwen3-14B analysis — structured JSON, NL explanation, or recommended action |
| **📝 SAR Draft** | Auto-generate FinCEN SAR narrative, fully editable before submission |
| **🕸 AML Graph** | D3.js force-directed transaction network with suspicious-edge highlighting |
| **[▶ Live Demo](https://youtu.be/YvvmbmW6nUs)** | One-click end-to-end walkthrough: card fraud / AML fan-out / legitimate TX |

> ¹ **AML GNN metrics context:** Standalone classification metrics are suppressed by extreme class imbalance (0.18% laundering rate across 5M transactions). The GNN's role is not standalone classification — it encodes multi-hop graph structure (fan-out rings, fan-in aggregators, cycle detection) that is invisible to the tabular LightGBM. The two models are used together: the LightGBM handles tabular velocity/amount signals while the GNN adds structural topology. Low standalone PR-AUC on a 1-in-565 positive class is expected and consistent with published results on the IBM HI-Small benchmark ([arXiv:2306.16424](https://arxiv.org/abs/2306.16424)).

---

## ⚡ Performance

| Path | Latency | Notes |
|---|---|---|
| Tier-1 Card Scorer (LightGBM) | **< 2 ms** | Single transaction, measured via `latency_ms` in API response |
| Tier-1 AML Scorer (LightGBM) | **< 2 ms** | Single transaction, measured via `latency_ms` in API response |
| Tier-2 LLM — fast mode (streaming) | **~3–6 s TTFT**, ~50–80 tok/s | vLLM on MI300X, bfloat16, max_tokens=800 |
| Tier-2 LLM — SAR draft (non-streaming) | **~8–15 s total** | max_tokens=1200, includes `latency_ms` + `tokens_per_sec` in response |
| System throughput (Tier-1 only) | **> 50K TPS** | LightGBM batch inference; Tier-2 invoked only for flagged transactions |

All API responses for scoring endpoints include `"latency_ms"` measured server-side. SAR/analyze responses additionally include `"tokens_generated"` and `"tokens_per_sec"` for observability.

---

## 🤗 Hugging Face Resources

| Resource | Type | Link |
|---|---|---|
| Tier-1 Scorers (LightGBM + GNN) | Model | [naazimsnh02/fraudsentinel-tier1-scorers](https://huggingface.co/naazimsnh02/fraudsentinel-tier1-scorers) |
| AML GNN (standalone) | Model | [naazimsnh02/fraudsentinel-aml-gnn](https://huggingface.co/naazimsnh02/fraudsentinel-aml-gnn) |
| Tier-2 LLM LoRA Adapter | Model | [naazimsnh02/fraudsentinel-qwen3-14b-lora](https://huggingface.co/naazimsnh02/fraudsentinel-qwen3-14b-lora) |
| Tier-2 LLM Merged (16-bit) | Model | [naazimsnh02/fraudsentinel-qwen3-14b-merged](https://huggingface.co/naazimsnh02/fraudsentinel-qwen3-14b-merged) |
| SFT Dataset | Dataset | [naazimsnh02/fraud-financial-crime-qwen3-sft-v2](https://huggingface.co/datasets/naazimsnh02/fraud-financial-crime-qwen3-sft-v2) |

---

## 📊 Dataset

The Tier-2 LLM is fine-tuned on **11,816 ChatML conversations** generated via **label-grounded synthesis** (no teacher LLM, no hallucinated labels):

- **Sources:** Sparkov credit card transactions (1.3M tx) + IBM AML HI-Small (5M tx)
- **Composition:** ~44% fraud/laundering, ~56% legitimate (incl. hard negatives)
- **Task mix:** structured JSON, explanations, recommendations, multi-turn HITL dialogue, scoring
- **Output schema:** risk score, risk level, typology, key signals, feature importance, recommended action, SAR rationale

See the [dataset card](https://huggingface.co/datasets/naazimsnh02/fraud-financial-crime-qwen3-sft-v2) for full details.

---

## 🚀 Deployment

See [`docs/DEPLOY.md`](docs/DEPLOY.md) for full instructions.

**Quick start:**
```bash
# One-time setup (installs deps + downloads Tier-1 model files)
bash scripts/install.sh

# Terminal 1 — inference engine
bash scripts/start.sh vllm

# Terminal 2 — app server
bash scripts/start.sh app
```

Open: `https://notebooks.amd.com/<your-pod>/proxy/7860/`

---

## 🔬 Research Foundations

| Reference | Finding |
|---|---|
| [arXiv:2507.14785](https://arxiv.org/abs/2507.14785) | LLMs are weak tabular classifiers but excel at explanation/reasoning |
| [arXiv:2306.16424](https://arxiv.org/abs/2306.16424) | IBM AML dataset — graph structure essential for laundering detection |
| [arXiv:2312.13896](https://arxiv.org/abs/2312.13896) | Boosted trees excel at CNP fraud with velocity features |
| [arXiv:2210.14360](https://arxiv.org/abs/2210.14360) | LaundroGraph — graph-based AML detection |
| [arXiv:2505.09388](https://arxiv.org/abs/2505.09388) | Qwen3 technical report |

---

## 🛠️ Tech Stack

- **Tier-1 ML:** LightGBM, PyTorch Geometric (GINEConv)
- **Tier-2 LLM:** Qwen3-14B, LoRA (r=16, α=32, all-linear), Unsloth 2026.6.1, TRL 0.22.2
- **Training Hardware:** AMD MI300X (192 GB VRAM), ROCm 7.0, PyTorch 2.10
- **Precision:** bfloat16
- **Serving:** vLLM 0.11.0rc2 (ROCm build)
- **Backend:** FastAPI + uvicorn
- **Frontend:** HTML/JS SPA with D3.js (served by FastAPI, no build step)

---

## ⚠️ Disclaimer

FraudSentinel is a **research-grade MVP** built on publicly available synthetic/semi-synthetic datasets (Sparkov, IBM HI-Small). It demonstrates a production-viable architecture and ML pipeline; the scoring thresholds and typology heuristics are illustrative and calibrated to these datasets. Productionizing for live customer adjudication requires independent validation on real transaction data, bias and fairness review, regulatory compliance checks, and human-in-the-loop controls appropriate to the jurisdiction. AI-generated SAR narratives are **drafts only** and must be reviewed by a qualified BSA/AML officer before filing.

---

## 📄 License

Apache 2.0 — base model (Qwen3) and LoRA fine-tuning recipe are Apache-2.0 licensed.
