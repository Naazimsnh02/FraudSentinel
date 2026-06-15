"""
Gradio 6 head tags and CSS for FraudSentinel.
Imported by dashboard.py and app.py.
"""

# Scripts/fonts loaded in <head> via Gradio 6 Blocks(head=...) parameter.
# Self-hosted assets or CDN-pinned versions for offline reliability.
HEAD_TAGS = """\
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
"""

# CSS injected via Gradio 6 Blocks(css=...) parameter.
# Targets .gradio-container to override Gradio's defaults, then defines
# FraudSentinel's own design system via CSS custom properties.
DARK_CSS = """\
/* ── Gradio 6 chrome reset ───────────────────────────────────────────────── */
.gradio-container {
  background: #080d18 !important;
  padding: 0 !important;
  margin: 0 !important;
  max-width: 100% !important;
  min-height: 100vh;
}
/* Gradio 6 fills the viewport when fill_height=True */
.gradio-container > .main > .wrap { padding: 0 !important; gap: 0 !important; }
footer { display: none !important; }

/* ── Design tokens ───────────────────────────────────────────────────────── */
:root {
  --bg0:    #080d18;
  --bg1:    #0e1525;
  --bg2:    #14203a;
  --bg3:    #1a2a4a;
  --border:        #1e3060;
  --border-bright: #2a4080;
  --accent:      #3b82f6;
  --accent-glow: #60a5fa;
  --green:  #22c55e;
  --yellow: #eab308;
  --orange: #f97316;
  --red:    #ef4444;
  --text:      #e2e8f0;
  --text-dim:  #94a3b8;
  --text-muted:#475569;
  --font:      'Inter', ui-sans-serif, system-ui, sans-serif;
  --font-mono: 'JetBrains Mono', 'Courier New', monospace;
  --radius:    8px;
  --radius-lg: 12px;
  --shadow:    0 4px 24px rgba(0,0,0,0.4);
}

/* ── SPA root ────────────────────────────────────────────────────────────── */
#fs-root {
  font-family: var(--font);
  color: var(--text);
  background: var(--bg0);
  width: 100%;
  min-height: 100vh;
}

/* ── Header ─────────────────────────────────────────────────────────────── */
.fs-header {
  background: var(--bg1);
  border-bottom: 1px solid var(--border);
  padding: 0 1.5rem;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  height: 54px;
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(8px);
}
.fs-logo {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--accent-glow);
  letter-spacing: -0.02em;
  white-space: nowrap;
}
.fs-logo span { color: var(--text-muted); font-weight: 400; font-size: 0.85rem; }

/* ── Nav tabs ────────────────────────────────────────────────────────────── */
.fs-nav { display: flex; gap: 0.15rem; margin-left: 1.5rem; }
.fs-nav button {
  background: none;
  border: none;
  color: var(--text-dim);
  padding: 0.35rem 0.9rem;
  border-radius: var(--radius);
  font-size: 0.85rem;
  cursor: pointer;
  font-family: var(--font);
  transition: color 0.15s, background 0.15s;
  white-space: nowrap;
}
.fs-nav button:hover  { color: var(--text); background: var(--bg2); }
.fs-nav button.active { color: var(--accent-glow); background: var(--bg2); font-weight: 500; }

/* ── Status indicators ───────────────────────────────────────────────────── */
.fs-status {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.72rem;
  color: var(--text-dim);
  flex-shrink: 0;
}
.fs-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
.fs-dot.ok   { background: var(--green); box-shadow: 0 0 5px var(--green); }
.fs-dot.err  { background: var(--red); }
.fs-dot.warn { background: var(--yellow); animation: pulse-warn 2s infinite; }
@keyframes pulse-warn {
  0%,100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* ── Pages ───────────────────────────────────────────────────────────────── */
.fs-page { display: none; padding: 1.25rem 1.5rem; }
.fs-page.active { display: block; }

/* ── Cards ───────────────────────────────────────────────────────────────── */
.fs-card {
  background: var(--bg1);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.1rem;
}
.fs-card-title {
  font-size: 0.7rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-muted);
  margin-bottom: 0.85rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

/* ── Grid layouts ────────────────────────────────────────────────────────── */
.fs-grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
.fs-grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.75rem; }
@media (max-width: 860px) {
  .fs-grid-2, .fs-grid-3 { grid-template-columns: 1fr; }
}

/* ── Form controls ───────────────────────────────────────────────────────── */
textarea, select {
  background: var(--bg2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 0.55rem 0.75rem !important;
  font-family: var(--font) !important;
  font-size: 0.85rem !important;
  width: 100%;
  box-sizing: border-box;
  resize: vertical;
  outline: none;
  transition: border-color 0.15s;
  line-height: 1.5;
}
textarea { font-family: var(--font-mono) !important; font-size: 0.82rem !important; }
textarea:focus, select:focus { border-color: var(--accent) !important; }
label {
  font-size: 0.77rem;
  color: var(--text-dim);
  display: block;
  margin-bottom: 0.3rem;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.fs-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  padding: 0.5rem 1.1rem;
  border-radius: var(--radius);
  font-size: 0.85rem;
  font-weight: 500;
  cursor: pointer;
  border: none;
  font-family: var(--font);
  transition: background 0.15s, border-color 0.15s, opacity 0.15s;
  white-space: nowrap;
}
.fs-btn-primary { background: var(--accent); color: #fff; }
.fs-btn-primary:hover:not(:disabled) { background: var(--accent-glow); }
.fs-btn-primary:disabled { background: var(--bg3); color: var(--text-muted); cursor: not-allowed; }
.fs-btn-ghost { background: var(--bg2); color: var(--text); border: 1px solid var(--border); }
.fs-btn-ghost:hover:not(:disabled) { border-color: var(--border-bright); }
.fs-btn-danger { background: #450a0a; color: #fca5a5; border: 1px solid #7f1d1d; }
.fs-btn-danger:hover { background: #7f1d1d; }
.fs-btn-sm { padding: 0.3rem 0.75rem; font-size: 0.78rem; }

/* ── Risk badges ─────────────────────────────────────────────────────────── */
.badge {
  display: inline-block;
  padding: 0.18rem 0.55rem;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}
.badge-LOW      { background: #14532d; color: #86efac; }
.badge-MEDIUM   { background: #713f12; color: #fde68a; }
.badge-HIGH     { background: #7c2d12; color: #fed7aa; }
.badge-CRITICAL {
  background: #450a0a;
  color: var(--red);
  box-shadow: 0 0 0 0 rgba(239,68,68,0.5);
  animation: pulse-critical 1.4s infinite;
}
@keyframes pulse-critical {
  0%,100% { box-shadow: 0 0 0 0 rgba(239,68,68,0.5); }
  50%     { box-shadow: 0 0 0 5px rgba(239,68,68,0); }
}

/* ── Score gauge ─────────────────────────────────────────────────────────── */
.score-gauge { text-align: center; padding: 0.5rem; }
.score-gauge .score-num { font-size: 2.6rem; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
.score-gauge .score-sub { font-size: 0.72rem; color: var(--text-dim); margin-top: 0.2rem; }

/* ── Progress bar ────────────────────────────────────────────────────────── */
.fs-progress { height: 5px; background: var(--bg3); border-radius: 3px; overflow: hidden; }
.fs-progress-bar { height: 100%; border-radius: 3px; transition: width 0.4s ease; }

/* ── Signal list ─────────────────────────────────────────────────────────── */
.signal-list { list-style: none; padding: 0; margin: 0; display: flex; flex-direction: column; gap: 0.35rem; }
.signal-list li {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--orange);
  border-radius: 5px;
  padding: 0.35rem 0.65rem;
  font-size: 0.8rem;
  color: var(--text-dim);
}

/* ── Stream box (LLM output) ─────────────────────────────────────────────── */
.stream-box {
  background: var(--bg0);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.85rem;
  min-height: 100px;
  font-size: 0.83rem;
  line-height: 1.65;
  white-space: pre-wrap;
  word-break: break-word;
  font-family: var(--font-mono);
  color: var(--text);
  overflow-y: auto;
}
.stream-box.loading { border-color: var(--accent); }
.stream-box.loading::after { content: "▋"; animation: blink 0.75s steps(1) infinite; }
@keyframes blink { 0%,49% { opacity: 1; } 50%,100% { opacity: 0; } }

/* ── SAR textarea ────────────────────────────────────────────────────────── */
.sar-out { min-height: 300px !important; }

/* ── AML graph ───────────────────────────────────────────────────────────── */
#aml-graph-svg {
  width: 100%; height: 460px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  display: block;
}
.glink  { stroke-opacity: 0.65; }
.gnode circle  { stroke-width: 1.5; cursor: grab; }
.gnode circle:active { cursor: grabbing; }
.gnode text { font-size: 9px; fill: var(--text-muted); pointer-events: none; }

/* ── Utility ─────────────────────────────────────────────────────────────── */
.mt1 { margin-top: 0.5rem; }
.mt2 { margin-top: 0.85rem; }
.mt3 { margin-top: 1.25rem; }
.row { display: flex; gap: 0.5rem; flex-wrap: wrap; align-items: center; }
.sec-label {
  font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.1em; color: var(--text-muted); margin: 0.85rem 0 0.4rem;
}
.hint { font-size: 0.73rem; color: var(--text-muted); margin-top: 0.35rem; }

/* ── Toast ───────────────────────────────────────────────────────────────── */
#fs-toast {
  position: fixed; bottom: 1.25rem; right: 1.25rem; z-index: 9999;
  background: var(--bg2); border: 1px solid var(--border);
  border-radius: var(--radius); padding: 0.65rem 1rem;
  font-size: 0.82rem; display: none; max-width: 300px;
  box-shadow: var(--shadow);
  transition: opacity 0.2s;
}
"""
