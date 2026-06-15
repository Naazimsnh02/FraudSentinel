"""
FraudSentinel — FastAPI server (no Gradio)
==========================================
Serves the SPA dashboard directly as an HTML response and exposes
all fraud/AML API endpoints.  Works behind JupyterHub's /proxy/7860/
prefix without any path-stripping hacks.

Run from repo root:
  cd backend
  python -m uvicorn app:app --host 0.0.0.0 --port 7860 --reload

Or via scripts/start.sh app

Access:
  https://notebooks.amd.com/<pod>/proxy/7860/
"""
from __future__ import annotations
import json, os, logging, time, re
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from tier1.card_scorer import CardScorer
from tier1.aml_scorer import AMLScorer

# ── Config ─────────────────────────────────────────────────────────────────
VLLM_BASE  = os.getenv("VLLM_BASE",  "http://localhost:8000/v1")
# Auto-detected from vLLM at startup; override with FRAUDSENTINEL_MODEL env var
MODEL_NAME = os.getenv("FRAUDSENTINEL_MODEL", "naazimsnh02/fraudsentinel-qwen3-14b-merged")
VLLM_KEY   = os.getenv("VLLM_KEY", "not-needed")
PORT       = int(os.getenv("PORT", "7860"))

SYSTEM_PROMPT = (
    "You are FraudSentinel, an expert fraud detection and AML investigation "
    "assistant. Analyze transactions precisely, cite specific signals, and "
    "output well-structured responses."
)

log = logging.getLogger("fraudsentinel")
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s  %(name)s  %(message)s")

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)

def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text).lstrip()


class _ThinkStripper:
    """Buffers streaming SSE content to strip leading <think>…</think> blocks."""

    def __init__(self):
        self._buf = ""
        self._state = "probe"  # probe | in_think | passthrough

    def feed(self, chunk: str) -> str:
        if self._state == "passthrough":
            return chunk
        self._buf += chunk
        if self._state == "probe":
            if self._buf.startswith("<think>"):
                self._state = "in_think"
            elif len(self._buf) >= 7:
                self._state = "passthrough"
                out, self._buf = self._buf, ""
                return out
            return ""
        # in_think: wait for closing tag
        end = self._buf.find("</think>")
        if end != -1:
            self._state = "passthrough"
            out = self._buf[end + 8:].lstrip("\n")
            self._buf = ""
            return out
        return ""

# ── Global scorer instances ─────────────────────────────────────────────────
card_scorer: CardScorer | None = None
aml_scorer:  AMLScorer  | None = None


@asynccontextmanager
async def lifespan(api: FastAPI):
    global card_scorer, aml_scorer, MODEL_NAME
    log.info("Loading Tier-1 scorers…")
    try:
        card_scorer = CardScorer()
        log.info("  ✓ Card LightGBM loaded")
    except Exception as exc:
        log.warning("  ✗ Card scorer failed: %s", exc)
    try:
        aml_scorer = AMLScorer()
        log.info("  ✓ AML  LightGBM loaded")
    except Exception as exc:
        log.warning("  ✗ AML scorer failed: %s", exc)
    # Auto-detect the model ID vLLM actually registered (avoids HF-ID vs local-path mismatch)
    if not os.getenv("FRAUDSENTINEL_MODEL"):
        try:
            async with httpx.AsyncClient(timeout=4.0) as client:
                r = await client.get(f"{VLLM_BASE}/models")
                if r.status_code == 200:
                    models = r.json().get("data", [])
                    if models:
                        MODEL_NAME = models[0]["id"]
                        log.info("  ✓ vLLM model detected: %s", MODEL_NAME)
        except Exception:
            log.info("  ℹ vLLM not yet reachable; using default model name: %s", MODEL_NAME)
    log.info("Ready.")
    yield
    log.info("Shutdown.")


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="FraudSentinel API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dashboard HTML ───────────────────────────────────────────────────────────
def _build_page() -> str:
    from static.styles import DARK_CSS, HEAD_TAGS
    from static.dashboard import DASHBOARD_HTML
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FraudSentinel</title>
{HEAD_TAGS}
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #080d18; }}
{DARK_CSS}
</style>
</head>
<body>
{DASHBOARD_HTML}
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
@app.get("/ui/", response_class=HTMLResponse)
async def dashboard():
    return _build_page()


# ── /api/health ─────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    vllm_ok = False
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            r = await client.get(f"{VLLM_BASE}/models")
            vllm_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "card_scorer": card_scorer is not None,
        "aml_scorer":  aml_scorer  is not None,
        "vllm":        vllm_ok,
        "model":       MODEL_NAME,
    }


# ── /api/score/card ─────────────────────────────────────────────────────────
@app.post("/api/score/card")
async def score_card(request: Request):
    if card_scorer is None:
        return JSONResponse({"error": "Card scorer not loaded"}, status_code=503)
    tx = await request.json()
    t0 = time.perf_counter()
    result = card_scorer.score(tx)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return JSONResponse(result)


# ── /api/score/aml ──────────────────────────────────────────────────────────
@app.post("/api/score/aml")
async def score_aml(request: Request):
    if aml_scorer is None:
        return JSONResponse({"error": "AML scorer not loaded"}, status_code=503)
    tx = await request.json()
    t0 = time.perf_counter()
    result = aml_scorer.score(tx)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return JSONResponse(result)


# ── /api/analyze/stream (SSE) ───────────────────────────────────────────────
@app.post("/api/analyze/stream")
async def analyze_stream(request: Request):
    body      = await request.json()
    msgs      = body.get("messages", [])
    mode      = body.get("mode", "fast")
    max_tok   = body.get("max_tokens", 800)
    temp      = 0.1 if mode == "fast" else 0.3
    messages  = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs

    async def sse() -> AsyncGenerator[bytes, None]:
        stripper = _ThinkStripper()
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST", f"{VLLM_BASE}/chat/completions",
                    headers={"Authorization": f"Bearer {VLLM_KEY}"},
                    json={"model": MODEL_NAME, "messages": messages,
                          "stream": True, "temperature": temp,
                          "max_tokens": max_tok},
                ) as resp:
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload.strip() == "[DONE]":
                            yield b"data: [DONE]\n\n"
                            break
                        try:
                            chunk = json.loads(payload)
                            delta = chunk["choices"][0].get("delta", {})
                            raw = delta.get("content", "")
                            if raw:
                                visible = stripper.feed(raw)
                                if not visible:
                                    continue
                                delta["content"] = visible
                                chunk["choices"][0]["delta"] = delta
                            yield f"data: {json.dumps(chunk)}\n\n".encode()
                        except Exception:
                            yield f"data: {payload}\n\n".encode()
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode()

    return StreamingResponse(
        sse(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /api/analyze (non-streaming, for SAR) ───────────────────────────────────
@app.post("/api/analyze")
async def analyze(request: Request):
    body    = await request.json()
    msgs    = body.get("messages", [])
    max_tok = body.get("max_tokens", 1200)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{VLLM_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {VLLM_KEY}"},
            json={"model": MODEL_NAME, "messages": messages,
                  "stream": False, "temperature": 0.1, "max_tokens": max_tok},
        )
    data = resp.json()
    latency_ms = round((time.perf_counter() - t0) * 1000)
    usage = data.get("usage", {})
    content = _strip_think(data["choices"][0]["message"]["content"])
    return JSONResponse({
        "content": content,
        "latency_ms": latency_ms,
        "tokens_generated": usage.get("completion_tokens"),
        "tokens_per_sec": round(usage.get("completion_tokens", 0) / (latency_ms / 1000), 1) if latency_ms > 0 else None,
    })


# ── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False,
                log_level="info")
