"""
FraudSentinel dashboard — HTML + JS SPA.
Imports styles from static/styles.py to stay within file size limits.
"""
from static.styles import DARK_CSS, HEAD_TAGS  # re-export for app.py

# ─────────────────────────────────────────────────────────────────────────────
# Full single-page application injected via Gradio 6's gr.HTML layout element.
# All API calls go to FastAPI /api/* endpoints via fetch().
# Streaming uses the browser's ReadableStream / SSE reader directly.
# D3.js is loaded in HEAD_TAGS for the AML graph tab.
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""
<div id="fs-root">

<!-- ══════════ HEADER ══════════ -->
<header class="fs-header">
  <div class="fs-logo">🛡 FraudSentinel <span>/ AMD MI300X · Qwen3-14B</span></div>
  <nav class="fs-nav" id="fs-nav">
    <button class="active" data-page="score">⚡ Score</button>
    <button data-page="analyze">🧠 Analyze</button>
    <button data-page="sar">📝 SAR Draft</button>
    <button data-page="graph">🕸 AML Graph</button>
    <button data-page="demo">▶ Live Demo</button>
  </nav>
  <div class="fs-status">
    <span id="dot-vllm" class="fs-dot warn"></span>
    <span id="lbl-vllm" style="min-width:3.5rem">vLLM…</span>
    <span id="dot-card" class="fs-dot warn"></span><span>Card</span>
    <span id="dot-aml"  class="fs-dot warn"></span><span>AML</span>
  </div>
</header>

<!-- ══════════ PAGE: SCORE ══════════ -->
<div id="page-score" class="fs-page active">
  <div class="fs-grid-2" style="align-items:start">

    <div class="fs-card">
      <div class="fs-card-title">Transaction Input</div>
      <div class="row" style="margin-bottom:0.65rem">
        <button class="fs-btn fs-btn-ghost fs-btn-sm" onclick="loadSample('card')">💳 Sample Card</button>
        <button class="fs-btn fs-btn-ghost fs-btn-sm" onclick="loadSample('aml')">🏦 Sample AML</button>
        <button class="fs-btn fs-btn-ghost fs-btn-sm" onclick="loadSample('legit')">✅ Legit TX</button>
      </div>
      <label>Transaction JSON</label>
      <textarea id="tx-input" rows="15"
        placeholder='{"amt": 828.62, "category": "misc_net", "hour": 2, ...}'></textarea>
      <div class="row mt2">
        <button class="fs-btn fs-btn-ghost fs-btn-sm" id="btn-domain-card"
                onclick="setDomain('card')" style="border-color:var(--accent)">
          Card Fraud
        </button>
        <button class="fs-btn fs-btn-ghost fs-btn-sm" id="btn-domain-aml"
                onclick="setDomain('aml')">
          AML
        </button>
        <button class="fs-btn fs-btn-primary" onclick="runScore()" id="btn-score"
                style="margin-left:auto">
          ▶ Score
        </button>
      </div>
    </div>

    <div id="score-result-col">
      <div class="fs-card" id="score-empty" style="text-align:center;padding:3rem 1rem;color:var(--text-muted)">
        Score a transaction to see Tier-1 results.
      </div>
      <div class="fs-card" id="score-result" style="display:none">
        <div class="fs-card-title">Tier-1 Result</div>
        <div class="fs-grid-2" style="gap:1.25rem;align-items:center;margin-bottom:0.85rem">
          <div class="score-gauge">
            <div class="score-num" id="r-score">—</div>
            <div class="score-sub">Risk Score</div>
          </div>
          <div>
            <div id="r-badge" style="margin-bottom:0.5rem"></div>
            <div class="fs-progress" style="margin-bottom:0.6rem">
              <div class="fs-progress-bar" id="r-bar" style="width:0%"></div>
            </div>
            <div style="font-size:0.78rem;color:var(--text-dim)">
              Route to LLM: <strong id="r-route">—</strong>
            </div>
          </div>
        </div>
        <div class="sec-label">Top Signals</div>
        <ul class="signal-list" id="r-signals"></ul>
        <button class="fs-btn fs-btn-primary mt3" id="btn-deep" onclick="goAnalyze()"
                style="display:none;width:100%">
          🧠 Deep Analyze with Qwen3-14B →
        </button>
      </div>
    </div>
  </div>
</div>

<!-- ══════════ PAGE: ANALYZE ══════════ -->
<div id="page-analyze" class="fs-page">
  <div class="fs-grid-2" style="align-items:start">

    <div class="fs-card">
      <div class="fs-card-title">Case Input</div>
      <label>Paste transaction details, Tier-1 result, or investigator notes</label>
      <textarea id="ana-input" rows="11"
        placeholder="Transaction details and Tier-1 score will be pre-filled from the Score tab…"></textarea>
      <div class="row mt2">
        <select id="ana-task" style="flex:1;min-width:0">
          <option value="structured">Structured JSON output</option>
          <option value="explain">Natural language explanation</option>
          <option value="recommend">Recommended action</option>
          <option value="multiturn">Investigator dialogue</option>
        </select>
        <select id="ana-mode" style="width:9rem;flex-shrink:0">
          <option value="fast">⚡ Fast mode</option>
          <option value="deep">🔍 Deep Analysis (CoT)</option>
        </select>
      </div>
      <button class="fs-btn fs-btn-primary mt2" onclick="runAnalyze()" id="btn-analyze"
              style="width:100%">
        🧠 Analyze with Qwen3-14B
      </button>
    </div>

    <div class="fs-card">
      <div class="fs-card-title">
        LLM Output
        <span id="ana-mode-lbl" style="color:var(--accent-glow);font-weight:400;font-size:0.7rem"></span>
      </div>
      <div id="ana-output" class="stream-box" style="min-height:320px"></div>
      <div class="row mt2">
        <button class="fs-btn fs-btn-ghost fs-btn-sm" onclick="copyEl('ana-output')">Copy</button>
        <button class="fs-btn fs-btn-ghost fs-btn-sm" onclick="toSAR()">→ SAR Draft</button>
        <button class="fs-btn fs-btn-danger fs-btn-sm" onclick="clearEl('ana-output')"
                style="margin-left:auto">Clear</button>
      </div>
    </div>

  </div>
</div>

<!-- ══════════ PAGE: SAR DRAFT ══════════ -->
<div id="page-sar" class="fs-page">
  <div class="fs-grid-2" style="align-items:start">

    <div class="fs-card">
      <div class="fs-card-title">SAR Generation Input</div>
      <label>Case summary, transaction details, or analysis output</label>
      <textarea id="sar-input" rows="12"
        placeholder="Paste transaction details, risk analysis, or any relevant case information…"></textarea>
      <button class="fs-btn fs-btn-primary mt2" onclick="runSAR()" id="btn-sar"
              style="width:100%">
        📝 Generate SAR Draft
      </button>
      <div class="hint mt1">
        ⚠ AI-generated draft only. Requires human review and edit before filing with FinCEN.
      </div>
    </div>

    <div class="fs-card">
      <div class="fs-card-title">
        SAR Narrative Draft
        <span style="color:var(--yellow);font-weight:400">Human review required</span>
      </div>
      <textarea id="sar-output" class="sar-out" rows="14"
        placeholder="SAR narrative will appear here. Edit before submission."></textarea>
      <div class="row mt2">
        <button class="fs-btn fs-btn-ghost fs-btn-sm" onclick="copyEl('sar-output')">Copy</button>
        <button class="fs-btn fs-btn-ghost fs-btn-sm" onclick="clearEl('sar-output')">Clear</button>
      </div>
    </div>

  </div>
</div>

<!-- ══════════ PAGE: AML GRAPH ══════════ -->
<div id="page-graph" class="fs-page">
  <div class="fs-grid-2" style="align-items:start">

    <div class="fs-card">
      <div class="fs-card-title">Graph Edge List</div>
      <label>JSON array of {source, target, amount, is_suspicious}</label>
      <textarea id="graph-input" rows="16" placeholder='[
  {"source":"ACCT-001","target":"ACCT-002","amount":95000,"is_suspicious":true},
  {"source":"ACCT-001","target":"ACCT-003","amount":48000,"is_suspicious":false}
]'></textarea>
      <div class="row mt2">
        <button class="fs-btn fs-btn-ghost fs-btn-sm" onclick="loadGraphSample()">
          Load Fan-Out Sample
        </button>
        <button class="fs-btn fs-btn-primary" onclick="renderGraph()"
                style="margin-left:auto">
          🕸 Render
        </button>
      </div>
    </div>

    <div class="fs-card">
      <div class="fs-card-title">
        Transaction Network
        <span id="graph-stats" style="color:var(--text-dim);font-weight:400"></span>
      </div>
      <svg id="aml-graph-svg"></svg>
      <div class="row mt1" style="font-size:0.73rem;color:var(--text-dim)">
        <span>🔴 Suspicious</span>
        <span>🔵 Normal</span>
        <span style="margin-left:auto;color:var(--text-muted)">Drag nodes to explore</span>
      </div>
    </div>

  </div>
</div>

<!-- ══════════ PAGE: LIVE DEMO ══════════ -->
<div id="page-demo" class="fs-page">
  <div style="max-width:860px">
    <div class="fs-card">
      <div class="fs-card-title">End-to-End Walkthrough</div>
      <p style="color:var(--text-dim);font-size:0.88rem;margin:0 0 1.1rem">
        Pre-loaded patterns from the Sparkov and IBM AML datasets.
        Each scenario runs the full two-tier pipeline — Tier-1 scorer then Qwen3-14B analysis.
      </p>
      <div class="fs-grid-3" style="margin-bottom:1.1rem">
        <button class="fs-btn fs-btn-ghost" onclick="runDemo('fraud')"
                style="flex-direction:column;align-items:flex-start;padding:0.85rem;height:auto">
          <span style="font-weight:600;margin-bottom:0.2rem">💳 Card Fraud</span>
          <span style="font-size:0.73rem;color:var(--text-dim)">2AM misc_net, 2.1×p95, velocity spike</span>
        </button>
        <button class="fs-btn fs-btn-ghost" onclick="runDemo('aml')"
                style="flex-direction:column;align-items:flex-start;padding:0.85rem;height:auto">
          <span style="font-weight:600;margin-bottom:0.2rem">🕸 AML Fan-Out</span>
          <span style="font-size:0.73rem;color:var(--text-dim)">High out-degree, cross-currency, ACH</span>
        </button>
        <button class="fs-btn fs-btn-ghost" onclick="runDemo('legit')"
                style="flex-direction:column;align-items:flex-start;padding:0.85rem;height:auto">
          <span style="font-weight:600;margin-bottom:0.2rem">✅ Legitimate TX</span>
          <span style="font-size:0.73rem;color:var(--text-dim)">Normal purchase — auto-approved</span>
        </button>
      </div>
      <div id="demo-log" class="stream-box" style="min-height:380px"></div>
    </div>
  </div>
</div>

<!-- TOAST -->
<div id="fs-toast"></div>

</div><!-- #fs-root -->

<script>
/* ═══════════════════════ STATE ════════════════════════════════════════════ */
let _domain    = 'card';
let _lastPrompt = '';

// Detect the proxy base path from the current URL so fetch() calls work
// behind JupyterHub's /proxy/7860/ prefix.
// e.g.  https://notebooks.amd.com/<pod>/proxy/7860/
//   →   API_BASE = https://notebooks.amd.com/<pod>/proxy/7860
// Plain  http://localhost:7860/  →  API_BASE = http://localhost:7860
const _loc = window.location;
const _proxyMatch = _loc.pathname.match(/^(.*\/proxy\/\d+)/);
const _uiIdx = _loc.pathname.indexOf('/ui');
const API_BASE = _proxyMatch
  ? _loc.origin + _proxyMatch[1]                    // behind JupyterHub proxy
  : _uiIdx > 0
    ? _loc.origin + _loc.pathname.slice(0, _uiIdx)  // legacy /ui suffix
    : _loc.origin;                                   // dev: no prefix

const TASK_INSTRUCTIONS = {
  structured: 'Output a complete structured JSON risk assessment with these fields:\nrisk_score (0.0-1.0), risk_level (LOW/MEDIUM/HIGH/CRITICAL), conclusion, primary_typology, secondary_typology, key_signals (array), explanation, feature_importance (object, values sum to 1.0), recommended_action (one of: AUTO_APPROVE/APPROVE_WITH_MONITORING/STEP_UP_AUTH/TEMPORARY_HOLD/AUTO_BLOCK/SAR_REVIEW), sar_required (bool), sar_rationale.',
  explain:    'Write a clear, concise natural-language alert explanation for a fraud investigator. Cite specific transaction features. End with a recommended action.',
  recommend:  'State the recommended action from the 6-level taxonomy: AUTO_APPROVE / APPROVE_WITH_MONITORING / STEP_UP_AUTH / TEMPORARY_HOLD / AUTO_BLOCK / SAR_REVIEW. Justify it in 2-3 sentences.',
  multiturn:  'You are in an investigator dialogue session. Answer the investigator\'s question directly and precisely.',
};

/* ═══════════════════════ SAMPLES ══════════════════════════════════════════ */
const SAMPLES = {
  card: {
    amt: 828.62, category: 'misc_net', hour: 2, dow: 3,
    lat: 40.7128, long: -74.006, merch_lat: 34.0522, merch_lon: -118.2437,
    city_pop: 8336817, dob_year: 1988, gender: 'F', state: 'NY',
    tx_24h: 6, amt_24h: 1240.50, tx_1h: 3, mins_since_last: 4.2,
    amt_to_p95: 2.16
  },
  aml: {
    account: 'A0019283', account_1: 'B0058291',
    amount_paid: 95000, amount_received: 94840,
    payment_currency: 'USD', receiving_currency: 'EUR',
    payment_format: 'ACH', from_bank: 'BANKX', to_bank: 'BANKY',
    hour: 14, dow: 1,
    snd_out_deg: 47, snd_in_deg: 2, rcv_in_deg: 52
  },
  legit: {
    amt: 42.50, category: 'gas_transport', hour: 11, dow: 2,
    lat: 37.7749, long: -122.4194, merch_lat: 37.775, merch_lon: -122.418,
    city_pop: 883305, dob_year: 1975, gender: 'M', state: 'CA',
    tx_24h: 1, amt_24h: 42.50, tx_1h: 1, mins_since_last: 1440,
    amt_to_p95: 0.62
  },
};

/* ═══════════════════════ NAVIGATION ════════════════════════════════════════ */
document.getElementById('fs-nav').addEventListener('click', e => {
  const btn = e.target.closest('button[data-page]');
  if (!btn) return;
  document.querySelectorAll('.fs-page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('#fs-nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('page-' + btn.dataset.page).classList.add('active');
  btn.classList.add('active');
});

function gotoPage(name) {
  const btn = document.querySelector(`#fs-nav button[data-page="${name}"]`);
  if (btn) btn.click();
}

/* ═══════════════════════ HEALTH CHECK ══════════════════════════════════════ */
async function checkHealth() {
  try {
    const r = await fetch(API_BASE + '/api/health');
    const d = await r.json();
    dot('dot-vllm', d.vllm);
    dot('dot-card', d.card_scorer);
    dot('dot-aml',  d.aml_scorer);
    document.getElementById('lbl-vllm').textContent =
      d.vllm ? 'vLLM ✓' : 'vLLM ✗';
  } catch (_) {}
}
function dot(id, ok) {
  document.getElementById(id).className = 'fs-dot ' + (ok ? 'ok' : 'err');
}
checkHealth();
setInterval(checkHealth, 12000);

/* ═══════════════════════ DOMAIN SELECTOR ════════════════════════════════════ */
function setDomain(d) {
  _domain = d;
  document.getElementById('btn-domain-card').style.borderColor =
    d === 'card' ? 'var(--accent)' : 'var(--border)';
  document.getElementById('btn-domain-aml').style.borderColor =
    d === 'aml' ? 'var(--accent)' : 'var(--border)';
}

function loadSample(name) {
  const key = name === 'card' ? 'card' : name === 'aml' ? 'aml' : 'legit';
  document.getElementById('tx-input').value = JSON.stringify(SAMPLES[key], null, 2);
  setDomain(name === 'aml' ? 'aml' : 'card');
}

/* ═══════════════════════ TIER-1 SCORING ════════════════════════════════════ */
async function runScore() {
  const raw = document.getElementById('tx-input').value.trim();
  if (!raw) return toast('Paste a transaction JSON first', 'warn');
  let tx;
  try { tx = JSON.parse(raw); }
  catch (e) { return toast('Invalid JSON: ' + e.message, 'err'); }

  const btn = document.getElementById('btn-score');
  btn.disabled = true; btn.textContent = '⏳ Scoring…';

  try {
    const url = _domain === 'card'
      ? API_BASE + '/api/score/card'
      : API_BASE + '/api/score/aml';
    const r = await fetch(url, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(tx),
    });
    const d = await r.json();
    if (d.error) { toast(d.error, 'err'); return; }
    renderScore(d);
    _lastPrompt = buildPrompt(tx, d);
  } catch (e) {
    toast('Scoring failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false; btn.textContent = '▶ Score';
  }
}

function renderScore(d) {
  document.getElementById('score-empty').style.display = 'none';
  const el = document.getElementById('score-result');
  el.style.display = 'block';

  const pct = Math.min(100, Math.round(d.risk_score * 100));
  const col = riskColor(d.risk_level);

  document.getElementById('r-score').textContent   = d.risk_score.toFixed(4);
  document.getElementById('r-score').style.color   = col;
  document.getElementById('r-badge').innerHTML     =
    `<span class="badge badge-${d.risk_level}">${d.risk_level}</span>`;
  document.getElementById('r-bar').style.width     = pct + '%';
  document.getElementById('r-bar').style.background = col;
  document.getElementById('r-route').textContent   =
    d.route_to_llm ? 'YES — routing to Tier-2' : 'NO — auto-approve';
  document.getElementById('r-route').style.color   =
    d.route_to_llm ? 'var(--orange)' : 'var(--green)';
  document.getElementById('r-signals').innerHTML   =
    (d.top_signals || []).map(s => `<li>${s}</li>`).join('');
  document.getElementById('btn-deep').style.display =
    d.route_to_llm ? 'flex' : 'none';
}

function riskColor(level) {
  return {CRITICAL:'var(--red)', HIGH:'var(--orange)',
          MEDIUM:'var(--yellow)', LOW:'var(--green)'}[level] || 'var(--text)';
}

function buildPrompt(tx, scored) {
  return `Transaction:\n${JSON.stringify(tx, null, 2)}\n\n` +
    `Tier-1 result:\n` +
    `  risk_score : ${scored.risk_score}\n` +
    `  risk_level : ${scored.risk_level}\n` +
    `  signals    : ${(scored.top_signals||[]).join('; ')}\n` +
    `  domain     : ${scored.domain}`;
}

function goAnalyze() {
  document.getElementById('ana-input').value = _lastPrompt +
    '\n\nProvide a full structured JSON risk assessment.';
  gotoPage('analyze');
  runAnalyze();
}

/* ═══════════════════════ TIER-2 ANALYZE (STREAMING) ════════════════════════ */
async function runAnalyze() {
  const inp  = document.getElementById('ana-input').value.trim();
  if (!inp) return toast('Enter case details first', 'warn');
  const task = document.getElementById('ana-task').value;
  const mode = document.getElementById('ana-mode').value;

  const btn = document.getElementById('btn-analyze');
  btn.disabled = true;
  const out = document.getElementById('ana-output');
  out.textContent = '';
  out.className = 'stream-box loading';
  document.getElementById('ana-mode-lbl').textContent =
    mode === 'deep' ? '🔍 Chain-of-Thought' : '⚡ Fast';

  const content = TASK_INSTRUCTIONS[task] + '\n\n' + inp;

  try {
    await streamTo(out, [{role:'user', content}], mode, 900);
  } finally {
    out.className = 'stream-box';
    btn.disabled = false;
  }
}

/* ═══════════════════════ SAR DRAFT ═════════════════════════════════════════ */
async function runSAR() {
  const inp = document.getElementById('sar-input').value.trim();
  if (!inp) return toast('Enter case details first', 'warn');

  const btn = document.getElementById('btn-sar');
  btn.disabled = true; btn.textContent = '⏳ Generating…';
  document.getElementById('sar-output').value = '';

  const prompt = `Draft a FinCEN-compliant Suspicious Activity Report narrative for the following case.

Case details:
${inp}

Include these sections:
1. Summary of Suspicious Activity (2-3 sentences)
2. Subject and Account Information
3. Description of Suspicious Transactions (dates, amounts, methods)
4. Basis for Suspicion (specific indicators observed)
5. Prior SAR History (state "None known" if not provided)
6. Recommended Follow-Up Actions

Write in formal compliance language. Be specific and factual.`;

  try {
    const r = await fetch(API_BASE + '/api/analyze', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({messages:[{role:'user',content:prompt}], max_tokens:1400}),
    });
    const d = await r.json();
    document.getElementById('sar-output').value = d.content || d.error || 'No response.';
  } catch(e) {
    document.getElementById('sar-output').value = 'Error: ' + e.message;
    toast('SAR generation failed', 'err');
  } finally {
    btn.disabled = false; btn.textContent = '📝 Generate SAR Draft';
  }
}

function toSAR() {
  document.getElementById('sar-input').value =
    document.getElementById('ana-output').textContent;
  gotoPage('sar');
}

/* ═══════════════════════ AML GRAPH (D3 v7) ══════════════════════════════════ */
const GRAPH_SAMPLE = [
  {source:'ACCT-001',target:'ACCT-002',amount:95000,is_suspicious:true},
  {source:'ACCT-001',target:'ACCT-003',amount:42000,is_suspicious:true},
  {source:'ACCT-001',target:'ACCT-004',amount:39000,is_suspicious:true},
  {source:'ACCT-001',target:'ACCT-007',amount:45000,is_suspicious:false},
  {source:'ACCT-003',target:'ACCT-005',amount:38500,is_suspicious:true},
  {source:'ACCT-004',target:'ACCT-005',amount:37000,is_suspicious:true},
  {source:'ACCT-005',target:'ACCT-006',amount:72000,is_suspicious:true},
  {source:'ACCT-007',target:'ACCT-008',amount:44000,is_suspicious:false},
  {source:'ACCT-002',target:'ACCT-009',amount:93000,is_suspicious:true},
  {source:'ACCT-009',target:'ACCT-006',amount:90000,is_suspicious:true},
];

function loadGraphSample() {
  document.getElementById('graph-input').value =
    JSON.stringify(GRAPH_SAMPLE, null, 2);
}

function renderGraph() {
  if (typeof d3 === 'undefined') return toast('D3.js not loaded yet', 'warn');
  const raw = document.getElementById('graph-input').value.trim();
  let edges;
  try { edges = JSON.parse(raw); }
  catch(e) { return toast('Invalid JSON: ' + e.message, 'err'); }

  const svg = d3.select('#aml-graph-svg');
  svg.selectAll('*').remove();

  const W = svg.node().clientWidth  || 560;
  const H = svg.node().clientHeight || 460;

  const nodeMap = new Map();
  edges.forEach(e => {
    if (!nodeMap.has(e.source)) nodeMap.set(e.source, {id:e.source, suspicious:false});
    if (!nodeMap.has(e.target)) nodeMap.set(e.target, {id:e.target, suspicious:false});
    if (e.is_suspicious) {
      nodeMap.get(e.source).suspicious = true;
      nodeMap.get(e.target).suspicious = true;
    }
  });
  const nodes = Array.from(nodeMap.values());
  const links = edges.map(e => ({...e}));

  document.getElementById('graph-stats').textContent =
    `${nodes.length} accounts · ${edges.length} transfers · ` +
    `${edges.filter(e=>e.is_suspicious).length} suspicious`;

  // Arrow markers
  const defs = svg.append('defs');
  ['normal','suspicious'].forEach(t => {
    defs.append('marker')
      .attr('id','arr-'+t).attr('viewBox','0 -5 10 10')
      .attr('refX', 20).attr('refY', 0)
      .attr('markerWidth', 5).attr('markerHeight', 5)
      .attr('orient','auto')
      .append('path').attr('d','M0,-5L10,0L0,5')
      .attr('fill', t==='suspicious' ? 'var(--red)' : 'var(--border-bright)');
  });

  const sim = d3.forceSimulation(nodes)
    .force('link', d3.forceLink(links).id(d=>d.id).distance(110))
    .force('charge', d3.forceManyBody().strength(-260))
    .force('center', d3.forceCenter(W/2, H/2))
    .force('collide', d3.forceCollide(26));

  const link = svg.append('g').selectAll('line').data(links).join('line')
    .attr('class','glink')
    .attr('stroke', d => d.is_suspicious ? 'var(--red)' : 'var(--border-bright)')
    .attr('stroke-width', d => Math.max(1, Math.log((d.amount||1000)/5000+1)*2))
    .attr('marker-end', d => `url(#arr-${d.is_suspicious?'suspicious':'normal'})`);

  const node = svg.append('g').selectAll('g').data(nodes).join('g')
    .attr('class','gnode')
    .call(d3.drag()
      .on('start',(e,d)=>{ if(!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
      .on('drag', (e,d)=>{ d.fx=e.x; d.fy=e.y; })
      .on('end',  (e,d)=>{ if(!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

  node.append('circle').attr('r', 15)
    .attr('fill', d => d.suspicious ? '#450a0a' : '#1e3a5f')
    .attr('stroke', d => d.suspicious ? 'var(--red)' : 'var(--accent)');

  node.append('text').attr('dy','0.35em').attr('text-anchor','middle')
    .text(d => d.id.slice(-5));

  // Amount labels on edges
  const edgeLbl = svg.append('g').selectAll('text').data(links).join('text')
    .attr('font-size','8px').attr('fill','var(--text-muted)')
    .attr('text-anchor','middle')
    .text(d => d.amount >= 1000
      ? '$' + (d.amount/1000).toFixed(0) + 'k'
      : '$' + d.amount);

  sim.on('tick', () => {
    link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y)
        .attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
    node.attr('transform',d=>`translate(${d.x},${d.y})`);
    edgeLbl.attr('x',d=>(d.source.x+d.target.x)/2)
            .attr('y',d=>(d.source.y+d.target.y)/2-4);
  });
}

/* ═══════════════════════ LIVE DEMO ══════════════════════════════════════════ */
async function runDemo(key) {
  const log = document.getElementById('demo-log');
  log.textContent = '';
  log.className = 'stream-box';

  const scenarios = {
    fraud: {label:'💳 Card Fraud',   domain:'card', tx:SAMPLES.card,  task:'structured'},
    aml:   {label:'🕸 AML Fan-Out', domain:'aml',  tx:SAMPLES.aml,   task:'explain'},
    legit: {label:'✅ Legit TX',     domain:'card', tx:SAMPLES.legit, task:'structured'},
  };
  const s = scenarios[key];

  function write(text, color) {
    const span = document.createElement('span');
    span.style.color = color || 'var(--text)';
    span.textContent = text + '\n';
    log.appendChild(span);
    log.scrollTop = log.scrollHeight;
  }

  write(`\n▶ Scenario: ${s.label}`, 'var(--accent-glow)');
  write('──────────────────────────────────────────────', 'var(--border-bright)');
  write(`\n[TIER-1] Scoring ${s.domain.toUpperCase()} transaction…`, 'var(--text-dim)');

  let scored;
  try {
    const r = await fetch(API_BASE + `/api/score/${s.domain}`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(s.tx),
    });
    scored = await r.json();
  } catch(e) {
    write(`✗ Tier-1 failed: ${e.message} — is the app running?`, 'var(--red)');
    return;
  }

  write(`\n  ✓ Tier-1 complete`, 'var(--green)');
  write(`    risk_score : ${scored.risk_score}`, 'var(--text)');
  write(`    risk_level : ${scored.risk_level}`,
        {CRITICAL:'var(--red)',HIGH:'var(--orange)',MEDIUM:'var(--yellow)',LOW:'var(--green)'}[scored.risk_level]);
  write(`    route_llm  : ${scored.route_to_llm}`,
        scored.route_to_llm ? 'var(--orange)' : 'var(--green)');
  if (scored.top_signals?.length)
    write(`    signals    : ${scored.top_signals.join(' | ')}`, 'var(--text-dim)');

  if (!scored.route_to_llm) {
    write(`\n✓ AUTO-APPROVE — below routing threshold. No LLM call needed.`, 'var(--green)');
    write(`\nDemo complete.\n`, 'var(--accent-glow)');
    return;
  }

  write(`\n[TIER-2] Routing to Qwen3-14B (fast mode)…`, 'var(--text-dim)');
  write(`  Streaming response:\n`, 'var(--text-dim)');

  const prompt = buildPrompt(s.tx, scored) + '\n\n' +
    TASK_INSTRUCTIONS[s.task];

  const span = document.createElement('span');
  span.style.color = 'var(--text)';
  log.appendChild(span);

  try {
    const resp = await fetch(API_BASE + '/api/analyze/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages:[{role:'user',content:prompt}], mode:'fast'}),
    });
    const reader = resp.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const p = line.slice(6).trim();
        if (p === '[DONE]') break;
        try {
          const delta = JSON.parse(p).choices?.[0]?.delta?.content;
          if (delta) { span.textContent += delta; log.scrollTop = log.scrollHeight; }
        } catch(_){}
      }
    }
    write(`\n\n✓ Demo complete.`, 'var(--accent-glow)');
  } catch(e) {
    write(`\n✗ Tier-2 failed: ${e.message} — is vLLM running on port 8000?`, 'var(--red)');
  }
}

/* ═══════════════════════ SSE STREAMING HELPER ═══════════════════════════════ */
async function streamTo(el, messages, mode, maxTokens) {
  try {
    const resp = await fetch(API_BASE + '/api/analyze/stream', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({messages, mode, max_tokens: maxTokens}),
    });
    const reader = resp.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\n'); buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const p = line.slice(6).trim();
        if (p === '[DONE]') return;
        try {
          const delta = JSON.parse(p).choices?.[0]?.delta?.content;
          if (delta) { el.textContent += delta; el.scrollTop = el.scrollHeight; }
        } catch(_){}
      }
    }
  } catch(e) {
    el.textContent += '\n[Error: ' + e.message + ']';
    toast('Stream error: ' + e.message, 'err');
  }
}

/* ═══════════════════════ UTILITIES ══════════════════════════════════════════ */
function copyEl(id) {
  const el = document.getElementById(id);
  const text = el.value !== undefined ? el.value : el.textContent;
  navigator.clipboard.writeText(text).then(() => toast('Copied ✓', 'ok'));
}

function clearEl(id) {
  const el = document.getElementById(id);
  if (el.value !== undefined) el.value = ''; else el.textContent = '';
}

let _toastTimer;
function toast(msg, type) {
  const el = document.getElementById('fs-toast');
  el.textContent = msg;
  el.style.display = 'block';
  el.style.borderColor =
    type==='err' ? 'var(--red)' : type==='ok' ? 'var(--green)' : 'var(--yellow)';
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.style.display = 'none'; }, 3500);
}

/* ═══════════════════════ INIT ═══════════════════════════════════════════════ */
loadSample('card');  // pre-fill Score tab with card example
</script>
"""
