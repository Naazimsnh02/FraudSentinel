# FraudSentinel — Product Requirements Document (PRD)
## Enterprise Fraud Detection & Financial Crime Intelligence Platform

**Version:** 1.0  
**Date:** 2026-09-06  
**Author:** ML Engineering Team  
**Status:** In Development (Hackathon / Prototype → Production Pilot)  
**Namespace:** `naazimsnh02` (Hugging Face Hub)

---

## 1. Executive Summary

FraudSentinel is an **enterprise-grade, two-tier fraud detection and financial crime intelligence platform** designed to detect, prevent, and explain fraudulent activities across card-not-present (CNP) transactions and anti-money laundering (AML) scenarios.

The platform combines:
- **Tier-1:** Ultra-fast statistical models (LightGBM + Graph Neural Networks) that score 100% of the transaction stream in <10ms.
- **Tier-2:** A fine-tuned large language model (Qwen3) that provides explainable risk scoring, typology classification, recommended actions, and SAR (Suspicious Activity Report) drafting for flagged cases.
- **Human-in-the-Loop (HITL):** Investigator dashboard with case management, graph visualization, and audit trails.

**Key Differentiator:** Unlike pure LLM or pure statistical approaches, FraudSentinel uses the right tool for each job — fast triage at scale, deep reasoning and explanation for investigators, with full traceability and compliance support.

---

## 2. Problem Statement

### 2.1 Business Pain Points
1. **False Positive Fatigue:** Traditional rule-based systems generate excessive alerts, overwhelming investigators.
2. **Latency Constraints:** Enterprise payment processors require sub-50ms decisioning; LLMs alone take 2–4 seconds per transaction.
3. **Explainability Gap:** Black-box ML models fail regulatory scrutiny (GDPR Article 22, EU AI Act, FinCEN guidance).
4. **Graph Blindness:** Tabular models cannot detect multi-hop money laundering patterns (fan-out, gather-scatter, cycles).
5. **SAR Bottleneck:** Manual SAR drafting takes 2–4 hours per case, creating compliance backlogs.

### 2.2 Research-Validated Constraints
- LLMs are weak tabular classifiers (GPT-4o achieves only ~64% accuracy on AML benchmarks) but excel at explanation quality and structured reasoning (arXiv:2507.14785).
- A 14B-parameter LLM decodes ~40–80 tok/s; a 200-token response requires **2–4 seconds**, not <50ms (arXiv:2505.09388, Qwen3 technical report).
- Money laundering is inherently a **graph pattern**; single-transaction models max out at ROC-AUC ~0.82 regardless of architecture (arXiv:2306.16424, IBM NeurIPS 2023).

---

## 3. Solution Architecture

### 3.1 Two-Tier Design Pattern

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INCOMING TRANSACTION STREAM                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 1: REAL-TIME TRIAGE (<10ms, 100% coverage)                            │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │ Card Scorer │    │ AML Scorer  │    │ AML GNN     │                      │
│  │ (LightGBM)  │    │ (LightGBM)  │    │ (Graph NN)  │                      │
│  │ PR-AUC 0.967│    │ Pre-filter  │    │ ROC-AUC 0.89│                      │
│  └─────────────┘    └─────────────┘    └─────────────┘                      │
│        │                  │                  │                               │
│        └──────────────────┴──────────────────┘                               │
│                           │                                                  │
│                    [Risk Score + Routing Decision]                           │
│                           │                                                  │
│              ┌────────────┴────────────┐                                     │
│              ▼                         ▼                                     │
│      [AUTO APPROVE]            [ROUTE TO TIER 2]                             │
│      <0.4% flagged             ~0.4–17% flagged                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  TIER 2: INTELLIGENT EXPLANATION (async, flagged cases only)                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Fine-Tuned Qwen3-8B/14B (LoRA Adapter)                             │    │
│  │  Apache-2.0, portable across GPU tiers                              │    │
│  │                                                                     │    │
│  │  • Structured Risk JSON (score, level, signals, typology)           │    │
│  │  • Natural Language Explanation (investigator-facing)               │    │
│  │  • Recommended Action (6-level decision matrix)                     │    │
│  │  • SAR Draft (auto-generated, human-reviewed)                       │    │
│  │  • Deep Analysis Mode (thinking/CoT, optional escalation)           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  HUMAN-IN-THE-LOOP                                                  │    │
│  │  Investigator Dashboard → Approve / Escalate / Modify SAR / Block   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Design Rationale

| Decision | Rationale | Research Basis |
|----------|-----------|----------------|
| Two-tier vs. single LLM | LLMs cannot score in <50ms; statistical models cannot explain | arXiv:2507.14785, 2210.14360 |
| LightGBM for cards | Tabular fraud (amount, velocity, geo) is well-captured by boosted trees | arXiv:2312.13896 |
| GNN for AML | Laundering is multi-hop graph pattern (fan-out, cycles) invisible to tabular models | arXiv:2306.16424 (IBM) |
| Qwen3-8B default / 14B premium | 8B fits single 24GB GPU (portable); 14B for premium tier; both Apache-2.0 | Qwen3 report (arXiv:2505.09388) |
| LoRA adapters | Same recipe (r=16, alpha=32, all-linear) ports across 8B/14B; vLLM hot-swaps per tenant | PEFT best practices |
| Thinking mode OFF default | Adds seconds of CoT tokens; expose as "Deep Analysis" escalation only | UX latency requirements |

---

## 4. Detailed Functional Requirements

### 4.1 Tier-1: Real-Time Scoring (FR-T1)

#### FR-T1.1 Card Fraud Scorer
- **Input:** Single transaction dict with fields: `amount`, `merchant`, `category`, `timestamp`, `card_id`, `zip`, `lat`, `long`, `dob`, `merch_lat`, `merch_long`
- **Output:** `{"risk_score": 0.0–1.0, "risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "route_to_llm": bool, "top_signals": [...]}`
- **Latency:** <10ms p99 on CPU
- **Throughput:** >10,000 TPS per core
- **Accuracy:** PR-AUC ≥0.95 on natural fraud rate (0.39%)
- **Status:** ✅ **IMPLEMENTED & VALIDATED**

#### FR-T1.2 AML Pre-Filter (Tabular Baseline)
- **Input:** Single transfer dict with fields: `amount_paid`, `amount_received`, `payment_currency`, `receiving_currency`, `payment_format`, `account`, `account_1`, `timestamp`
- **Output:** `{"risk_score": 0.0–1.0, "route_to_llm": bool}`
- **Latency:** <10ms p99 on CPU
- **Purpose:** High-recall pre-filter for low-resource deployments
- **Accuracy:** ROC-AUC 0.82 (expected baseline)
- **Status:** ✅ **IMPLEMENTED & VALIDATED**

#### FR-T1.3 AML GNN Scorer (Production)
- **Input:** Transaction multigraph (nodes=accounts, edges=transfers with edge features)
- **Output:** Per-edge risk score + graph-level anomaly flags
- **Latency:** <100ms for batch forward pass on GPU (acceptable for AML, which is batch-oriented)
- **Accuracy:** ROC-AUC ≥0.85, PR-AUC improvement over tabular baseline
- **Architecture:** GINE edge-classifier with edge features, temporal split, class-weighted loss
- **Status:** ✅ **IMPLEMENTED (v1 trained, v2 script ready)**

### 4.2 Tier-2: LLM Intelligence (FR-T2)

#### FR-T2.1 Structured Risk Scoring
- **Input:** Transaction(s) + contextual signals from Tier-1
- **Output:** Validated JSON schema:
  ```json
  {
    "risk_score": 0.0,
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "conclusion": "string",
    "primary_typology": "string",
    "secondary_typology": "string",
    "key_signals": ["string"],
    "explanation": "string",
    "feature_importance": {"signal": 0.0},
    "recommended_action": "AUTO_APPROVE|APPROVE_WITH_MONITORING|STEP_UP_AUTH|TEMPORARY_HOLD|AUTO_BLOCK|SAR_REVIEW",
    "sar_required": true,
    "sar_rationale": "string"
  }
  ```
- **Constraint:** Feature importance must sum to 1.0, values must be grounded in actual signals
- **Status:** 🔄 **PENDING** (dataset v2 ready, model training queued)

#### FR-T2.2 Explainable Alert Generation
- Natural language explanation tailored to investigator skill level
- Cites specific transaction features and graph patterns
- Multi-turn capability for follow-up questions
- **Status:** 🔄 **PENDING**

#### FR-T2.3 Recommended Actions
- 6-level decision matrix with confidence scores
- Must be overridden by human investigator
- Audit trail of AI recommendation vs. human decision
- **Status:** 🔄 **PENDING**

#### FR-T2.4 SAR Drafting
- Auto-generates FinCEN-compliant SAR narrative
- Populates structured fields (suspicious activity type, amounts, dates, involved parties)
- Human review and edit before submission
- **Status:** 🔄 **PENDING**

#### FR-T2.5 Deep Analysis Mode
- Optional Chain-of-Thought (CoT) reasoning for complex cases
- Triggered by investigator or automatic escalation rules
- Thinking mode enabled via Qwen3's thinking tokens
- **Status:** 🔄 **PENDING**

### 4.3 Human-in-the-Loop (FR-HITL)

#### FR-HITL.1 Investigator Dashboard
- Real-time alert queue with sorting, filtering, prioritization
- Case detail view: transaction timeline, risk signals, graph visualization
- One-click actions: Approve, Block, Escalate, Request More Info
- **Status:** 🔄 **PENDING**

#### FR-HITL.2 Case Management
- Case lifecycle: Open → Under Review → Resolved → Escalated → Closed
- Assignment to investigators with workload balancing
- SLA tracking (time-to-first-review, time-to-resolution)
- **Status:** 🔄 **PENDING**

#### FR-HITL.3 Feedback Loop
- Investigator corrections (overridden actions, edited SARs) logged as training data
- Periodic model retraining pipeline (weekly/monthly)
- A/B testing framework for model variants
- **Status:** 🔄 **PENDING**

### 4.4 Graph Visualization (FR-GRAPH)

#### FR-GRAPH.1 Transaction Network View
- Interactive graph of accounts and transactions
- Color-coded by risk score, edge thickness by amount
- Time-slider for temporal exploration
- Expand/collapse account neighborhoods
- **Status:** 🔄 **PENDING**

#### FR-GRAPH.2 Pattern Highlighting
- Auto-highlight suspicious subgraphs (cycles, fan-out, rapid pass-through)
- Overlay of known typologies (layering, structuring, smurfing)
- **Status:** 🔄 **PENDING**

---

## 5. Non-Functional Requirements

### 5.1 Performance
| Metric | Target | Current |
|--------|--------|---------|
| Tier-1 latency (p99) | <10ms | ✅ ~2-5ms (LightGBM) |
| Tier-1 throughput | >10K TPS | ✅ >50K TPS (LightGBM) |
| Tier-2 latency (p99) | <5s | 🔄 Not measured yet |
| Tier-2 throughput | >100 cases/min | 🔄 Not measured yet |
| End-to-end alert SLA | <2 min | 🔄 Not implemented |
| Graph batch forward | <500ms | ✅ ~100ms (GNN v1) |

### 5.2 Security & Compliance
- **GDPR Article 22:** Right to explanation — satisfied by structured JSON + NL explanation
- **EU AI Act:** High-risk system documentation — model cards, data sheets, risk assessments
- **FinCEN:** SAR filing requirements — structured fields + narrative draft
- **PCI-DSS:** Card data tokenization — no raw PANs in model inputs
- **SOC 2:** Audit trails for all decisions and model changes
- **Status:** 🔄 **Architecture defined, implementation pending**

### 5.3 Reliability
- Tier-1 fallback: if model service fails, route 100% to rule-based engine (fail-open for availability)
- Tier-2 fallback: if LLM service fails, queue for human review with Tier-1 score only
- Model versioning: immutable model artifacts with SHA-256 checksums
- **Status:** 🔄 **Pending**

### 5.4 Scalability
- Horizontal scaling of Tier-1 via stateless microservices
- Tier-2 via vLLM with continuous batching and LoRA adapter swapping
- Graph storage: Neptune/Neo4j for AML subgraph queries
- **Status:** 🔄 **Pending**

---

## 6. Data Architecture

### 6.1 Datasets

#### Dataset v2: LLM Training Data
- **Repo:** `naazimsnh02/fraud-financial-crime-qwen3-sft-v2`
- **Size:** 11,816 ChatML conversations
- **Format:** ChatML (messages column)
- **Composition:**
  - ~44% fraud/laundering cases
  - ~56% legitimate transactions (including hard negatives)
  - Tasks: structured JSON (~3.4K), explain (~3.4K), recommend, multiturn HITL, score
- **Sources:**
  - Card: `pointe77/credit-card-transaction` (Sparkov, 1.3M transactions, 0.58% fraud)
  - AML: `eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans` (5M transactions, 0.10% laundering)
- **Generation Method:** Label-grounded (no teacher LLM) — real tabular data → deterministic signal extraction → structured prompt generation → ChatML output
- **Status:** ✅ **COMPLETE**

#### Dataset v1 (Legacy)
- **Repo:** `naazimsnh02/fraud-financial-crime-qwen3-sft`
- **Status:** Superseded by v2, kept for reference

### 6.2 Data Pipeline
```
Raw Transaction Stream
    │
    ├──> Feature Engineering (real-time)
    │    ├──> Card: velocity, geo-distance, time-of-day, category anomaly
    │    └──> AML: degree, amount ratios, currency mismatch, self-loop
    │
    ├──> Tier-1 Scoring
    │    ├──> Card: LightGBM → risk_score
    │    └──> AML: GNN → risk_score
    │
    └──> Tier-2 Routing (if risk_score > threshold)
         └──> Context Assembly → LLM Prompt → Qwen3 Inference
              └──> Structured JSON + Explanation + SAR Draft
```

### 6.3 Data Quality
- **Hard Negatives:** Legitimate transactions with strong structural flags (high-degree accounts, large amounts) included so model learns NOT to over-flag
- **Temporal Split:** Train/val/test sorted by time to prevent leakage
- **Class Balance:** Fraud/legit ~44/56 in training data (upsampled fraud for LLM learning; Tier-1 models use natural imbalance with class weights)

---

## 7. Model Architecture

### 7.1 Tier-1 Models

#### Card Scorer: LightGBM
- **Type:** Gradient Boosted Decision Trees
- **Features:** 20+ engineered signals (amount, velocity, geo, time, age, category)
- **Validation:** Temporal split, natural fraud rate (0.39%)
- **Metrics:** PR-AUC 0.967, ROC-AUC 0.999
- **Routing Threshold:** Recall 0.90 → Precision 0.964, Flagged 0.36%
- **Artifact:** `cc_lgbm_model.txt` + `cc_lgbm_preproc.joblib`
- **Status:** ✅ **COMPLETE**

#### AML Scorer: LightGBM (Pre-Filter)
- **Type:** Gradient Boosted Decision Trees
- **Features:** Amount, currency mismatch, payment format, time features
- **Validation:** Temporal split, natural laundering rate (0.18%)
- **Metrics:** ROC-AUC 0.82, PR-AUC 0.029
- **Purpose:** High-recall pre-filter or low-resource deployment
- **Artifact:** `aml_lgbm_model.txt` + `aml_lgbm_preproc.joblib`
- **Status:** ✅ **COMPLETE**

#### AML Scorer: GNN (Production)
- **Type:** Edge-classification Graph Neural Network
- **Architecture:** GINEConv with edge features
- **Node Features:** Log in-degree, log out-degree (computed on train edges only)
- **Edge Features:** log(amount_paid), log(amount_received), hour/23, day_of_week/6, currency_mismatch, self_loop, payment_format_onehot
- **Message Passing:** Bidirectional (forward + reverse edges) in v2 script
- **Training:** Temporal 70/10/20 split, class weight ~1244:1, Adam optimizer, 30-80 epochs
- **Validation:** Temporal test split (no graph leakage)
- **Metrics (v1 shipped):** ROC-AUC 0.894, PR-AUC 0.053
- **Metrics (v2 target):** ROC-AUC >0.90, PR-AUC >0.06 (with reverse edges + 3 layers)
- **Artifacts:** `aml_gnn.pt` (v1 trained), `train_gnn_aml.py` (v2 script)
- **Status:** ✅ **v1 TRAINED & SHIPPED; v2 SCRIPT READY**

### 7.2 Tier-2 Model

#### Base Model: Qwen3-8B / Qwen3-14B
- **License:** Apache-2.0 (verified)
- **Default:** Qwen3-8B (fits 24GB GPU, portable)
- **Premium:** Qwen3-14B (requires 40-48GB GPU)
- **Architecture:** Dense transformer (NOT MoE — avoids portability issues)

#### Fine-Tuning Method: LoRA
- **Config:** r=16, alpha=32, target_modules="all-linear"
- **Recipe:** SFT (Supervised Fine-Tuning) on ChatML format
- **Dataset:** `fraud-financial-crime-qwen3-sft-v2` (11,816 conversations)
- **Framework:** TRL + PEFT + Accelerate
- **Deployment:** vLLM with LoRA adapter hot-swapping per tenant
- **Status:** 🔄 **PENDING** (dataset ready, training queued for AMD MI300X)

#### Inference Modes
- **Fast Mode:** Thinking OFF, ~1-2s response, standard explanation
- **Deep Analysis:** Thinking ON (CoT), ~3-5s response, complex case reasoning

---

## 8. API Design

### 8.1 Tier-1 Scoring API
```python
POST /v1/score/card
Request:  {"amount": 123.45, "merchant": "AMAZON", "category": "shopping_net", ...}
Response: {"risk_score": 0.97, "risk_level": "CRITICAL", "route_to_llm": true,
           "top_signals": ["amount_3x_category_p95", "velocity_24h_spike"],
           "latency_ms": 3.2}

POST /v1/score/aml
Request:  {"account": "A123", "account_1": "B456", "amount_paid": 50000, ...}
Response: {"risk_score": 0.89, "route_to_llm": true,
           "graph_context": {"subgraph_size": 12, "cycle_detected": true},
           "latency_ms": 45.0}
```
**Status:** 🔄 **Partial (infer.py has scoring functions, no FastAPI wrapper yet)**

### 8.2 Tier-2 LLM API
```python
POST /v1/analyze
Request:  {"transaction_id": "txn_123", "tier1_score": 0.97,
           "signals": [...], "mode": "fast|deep"}
Response: {"structured": {...}, "explanation": "...", "recommended_action": "SAR_REVIEW",
           "sar_draft": "...", "confidence": 0.94, "latency_ms": 2100}

POST /v1/analyze/chat
Request:  {"case_id": "case_456", "message": "Why is this flagged?", "history": [...]}
Response: {"reply": "...", "structured_update": {...}}
```
**Status:** 🔄 **PENDING**

### 8.3 Case Management API
```python
GET  /v1/cases?status=open&priority=high&limit=50
POST /v1/cases/{id}/action   {"action": "approve|block|escalate", "notes": "..."}
GET  /v1/cases/{id}/timeline
POST /v1/cases/{id}/sar/submit
```
**Status:** 🔄 **PENDING**

---

## 9. UI/UX Requirements

### 9.1 Investigator Dashboard (React)
- **Alert Queue:** Sortable table with risk score, typology, age, assigned investigator
- **Case Detail:** Split pane — transaction details left, AI explanation right
- **Graph Viz:** Cytoscape.js or D3.js for interactive transaction networks
- **SAR Editor:** Rich text editor with AI-generated draft, human edit, compliance check
- **Actions:** One-click approve/block/escalate with mandatory notes for overrides
- **Mobile:** Responsive design for tablet use (not phone — investigators need screen real estate)
- **Status:** 🔄 **PENDING**

### 9.2 Admin Panel
- Model version management (rollback, canary deployment)
- Threshold tuning (risk score cutoffs per tenant/rule)
- Investigator performance metrics (cases/hour, accuracy vs. AI)
- Audit log viewer (searchable, exportable)
- **Status:** 🔄 **PENDING**

---

## 10. Infrastructure & DevOps

### 10.1 Deployment Architecture
```
┌────────────────────────────────────────────────────────────┐
│                        K8s Cluster                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ Tier-1 API  │  │ Tier-2 API  │  │ Dashboard (React)   │ │
│  │ (FastAPI)   │  │ (vLLM +     │  │ (Next.js)           │ │
│  │ 10 replicas │  │ LoRA) 2-4   │  │ CDN + S3            │ │
│  │             │  │ replicas    │  │                     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                                  │
│         └────────────────┼──────────────────────────────────┘
│                          │                                  │
│  ┌───────────────────────┴──────────────────────────────┐  │
│  │              Message Queue (Redis/RabbitMQ)           │  │
│  │   Tier-1 flags → Async Tier-2 analysis queue         │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                  │
│  ┌───────────────────────┴──────────────────────────────┐  │
│  │              Data Layer                               │  │
│  │  PostgreSQL (cases, users, audit)                     │  │
│  │  Neo4j/Neptune (transaction graph)                    │  │
│  │  S3 (model artifacts, SAR PDFs)                       │  │
│  └───────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### 10.2 Monitoring & Observability
- **MLflow:** Experiment tracking, model versioning, artifact storage
- **Prometheus + Grafana:** Latency histograms, throughput, error rates, GPU utilization
- **Trackio:** LLM training metrics, loss curves, evaluation dashboards
- **Alerting:** PagerDuty/Opsgenie for service degradation, model drift
- **Status:** 🔄 **PENDING**

### 10.3 CI/CD Pipeline
- GitHub Actions: lint, test, build containers
- Model registry: Hugging Face Hub (immutable model cards)
- Canary deployment: 5% traffic → 25% → 100% with automatic rollback on error spike
- **Status:** 🔄 **PENDING**

---

## 11. Security Requirements

### 11.1 Authentication & Authorization
- **RBAC:** Admin, Investigator, Auditor, System roles
- **SSO:** OAuth2/OIDC integration (Okta, Azure AD)
- **API Keys:** Scoped per tenant for Tier-1 scoring
- **Status:** 🔄 **PENDING**

### 11.2 Data Protection
- **Encryption:** At-rest (AES-256) and in-transit (TLS 1.3)
- **Tokenization:** Raw PANs replaced with tokens before model ingestion
- **PII Masking:** Automatic redaction in logs and model outputs
- **Status:** 🔄 **PENDING**

### 11.3 Audit Trail
- Every decision (AI + human) logged immutably
- Required fields: timestamp, user_id, case_id, action, rationale, model_version
- Retention: 7 years (regulatory requirement)
- **Status:** 🔄 **PENDING**

---

## 12. Completed Work ✅

| Component | Detail | Evidence |
|-----------|--------|----------|
| **Dataset v2** | 11,816 ChatML conversations, label-grounded, no teacher LLM hallucination | [HF Dataset](https://huggingface.co/datasets/naazimsnh02/fraud-financial-crime-qwen3-sft-v2) |
| **Card Scorer** | LightGBM, PR-AUC 0.967, ROC-AUC 0.999, routes <0.4% to LLM | `cc_lgbm_model.txt` + metrics |
| **AML Pre-Filter** | LightGBM baseline, ROC-AUC 0.82, recall-oriented | `aml_lgbm_model.txt` + metrics |
| **AML GNN v1** | Trained & shipped: ROC-AUC 0.894, PR-AUC 0.053 (beats tabular +83%) | `aml_gnn.pt` + `aml_gnn_metrics.json` |
| **GNN v2 Script** | Improved recipe: reverse edges, 3 layers, hidden 96, best-checkpoint, recall routing | `train_gnn_aml.py` |
| **Inference Helper** | `infer.py` loads models + preprocessors + routing thresholds | `infer.py` |
| **Model Cards** | Updated READMEs with exact metrics, architecture, limitations | Tier-1 repo + AML-GNN repo |
| **Research Validation** | Verified all pitch claims against real papers (no fabricated arXiv IDs) | arXiv:2507.14785, 2306.16424, 2312.13896, 2210.14360, 2505.09388 |
| **Architecture Correction** | Fixed original "14B LLM scores in <50ms" and "40% accuracy" claims to honest two-tier framing | Documented in this PRD |

---

## 13. Pending Work 🔄

### 13.1 Critical Path (Must Have for MVP)

| # | Task | Owner | Effort | Blockers |
|---|------|-------|--------|----------|
| 1 | **LLM Fine-Tuning** | User | 1-2 days | AMD MI300X access, TRL setup |
| 2 | **FastAPI Two-Tier Endpoint** | TBD | 4-6 hrs | LLM endpoint or mock |
| 3 | **React Dashboard (Alert Queue + Case Detail)** | TBD | 2-3 days | API endpoints |
| 4 | **Graph Visualization (Cytoscape.js)** | TBD | 1-2 days | Graph data API |
| 5 | **Case Management DB + API** | TBD | 1-2 days | Schema design |

### 13.2 High Priority (Should Have for Demo)

| # | Task | Owner | Effort |
|---|------|-------|--------|
| 6 | **SAR Drafting UI + Compliance Check** | TBD | 1-2 days |
| 7 | **HITL Feedback Loop (corrections → training data)** | TBD | 2-3 days |
| 8 | **RBAC + Auth (OAuth2)** | TBD | 1-2 days |
| 9 | **MLflow + Grafana Monitoring** | TBD | 1 day |
| 10 | **GNN v2 Training** (48GB GPU) | User | 2-3 hrs | GPU credits |

### 13.3 Medium Priority (Nice to Have)

| # | Task | Owner | Effort |
|---|------|-------|--------|
| 11 | **Multi-tenant LoRA adapter swapping** | TBD | 2-3 days |
| 12 | **Real-time streaming ingestion (Kafka/Kinesis)** | TBD | 2-3 days |
| 13 | **Model drift detection + auto-retraining** | TBD | 3-5 days |
| 14 | **Mobile-responsive investigator view** | TBD | 1-2 days |
| 15 | **A/B testing framework for model variants** | TBD | 2-3 days |

### 13.4 Low Priority (Future Roadmap)

| # | Task | Owner | Effort |
|---|------|-------|--------|
| 16 | **Multi-modal: document ingestion (PDF statements, IDs)** | TBD | 1-2 weeks |
| 17 | **Cross-border sanctions screening integration** | TBD | 1-2 weeks |
| 18 | **Federated learning for multi-bank deployments** | TBD | 2-4 weeks |
| 19 | **On-device edge scoring (mobile wallets)** | TBD | 2-3 weeks |

---

## 14. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| LLM fine-tuning OOM on MI300X | Medium | High | Start with Qwen3-8B LoRA (fits 24GB); use gradient checkpointing; scale to 14B only if 8B works |
| GNN v2 OOM on 24GB GPU | High (confirmed) | Medium | Run on 48GB+ GPU; v1 already shipped and functional |
| Synthetic data bias | Medium | High | Dataset v2 uses real tabular data + label-grounded generation; validate on holdout |
| Regulatory rejection of AI SARs | Medium | High | Every SAR requires human review + edit; AI is "draft assistant" not "filer" |
| Model drift in production | Medium | High | Feedback loop + periodic retraining + drift detection |
| Hackathon time constraints | High | Medium | Prioritize MVP (items 1-5 in pending); cut nice-to-haves |

---

## 15. Appendices

### 15.1 Hugging Face Hub Repositories

| Repo | Type | Purpose | URL |
|------|------|---------|-----|
| `fraudsentinel-tier1-scorers` | Model | All Tier-1 models + scripts + inference | [Link](https://huggingface.co/naazimsnh02/fraudsentinel-tier1-scorers) |
| `fraudsentinel-aml-gnn` | Model | Standalone GNN artifacts (v1) | [Link](https://huggingface.co/naazimsnh02/fraudsentinel-aml-gnn) |
| `fraud-financial-crime-qwen3-sft-v2` | Dataset | LLM training data (primary) | [Link](https://huggingface.co/datasets/naazimsnh02/fraud-financial-crime-qwen3-sft-v2) |
| `fraud-financial-crime-qwen3-sft` | Dataset | LLM training data (legacy v1) | [Link](https://huggingface.co/datasets/naazimsnh02/fraud-financial-crime-qwen3-sft) |

### 15.2 Source Datasets

| Dataset | Provider | Size | Fraud Rate | License |
|---------|----------|------|------------|---------|
| Credit Card Transactions | `pointe77/credit-card-transaction` | 1.3M | 0.58% | Unknown (research) |
| IBM AML HI-Small | `eexzzm/...` | 5M | 0.10% | Unknown (research) |

### 15.3 Verified Research References

| Citation | Topic | Key Finding |
|----------|-------|-------------|
| arXiv:2507.14785 | LLM for AML | LLMs weak at tabular classification, strong at explanation |
| arXiv:2306.16424 | IBM AML dataset | Multi-GNN reaches F1 60-70% on graph patterns |
| arXiv:2312.13896 | LightGBM card fraud | Boosted trees excel at CNP fraud with velocity features |
| arXiv:2210.14360 | LaundroGraph | Graph structure essential for laundering detection |
| arXiv:2505.09388 | Qwen3 report | 8B fits 24GB, 14B fits 48GB, Apache-2.0, thinking mode adds latency |

### 15.4 Hardware Requirements

| Component | Min | Recommended | Notes |
|-----------|-----|-------------|-------|
| Tier-1 Card (LightGBM) | 2 vCPU | 4 vCPU | CPU-only, <10ms |
| Tier-1 AML (GNN v1) | T4 16GB | A10G 24GB | GPU required |
| Tier-1 AML (GNN v2) | A10G 48GB | A100 80GB | Reverse edges + 3 layers |
| Tier-2 LLM (Qwen3-8B) | RTX 3090 24GB | A10G 24GB | LoRA adapter, vLLM |
| Tier-2 LLM (Qwen3-14B) | A100 40GB | A100 80GB | LoRA adapter, vLLM |
| Full Stack (dev) | 1× A10G | 2× A10G + CPU nodes | K8s cluster |

### 15.5 Glossary

| Term | Definition |
|------|------------|
| **SAR** | Suspicious Activity Report — filed with FinCEN for potential financial crimes |
| **HITL** | Human-in-the-Loop — human oversight and override of AI decisions |
| **LoRA** | Low-Rank Adaptation — parameter-efficient fine-tuning method |
| **GINE** | Graph Isomorphism Network with Edge features — message-passing GNN variant |
| **PR-AUC** | Area Under Precision-Recall Curve — key metric for imbalanced fraud detection |
| **ROC-AUC** | Area Under Receiver Operating Characteristic Curve — ranking quality metric |
| **CoT** | Chain-of-Thought — LLM reasoning mode that outputs intermediate steps |
| **vLLM** | High-throughput LLM inference engine with PagedAttention |

---

## 16. Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-09-06 | ML Engineering Team | Initial PRD compiled from all completed work and pending roadmap |

---

*End of Document*