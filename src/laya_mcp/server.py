"""MCP server for Laya, a non-autoregressive calibrated decision model.

Laya answers typed questions (choice / score / noul) about a state (text or JSON)
in a single forward pass and returns calibrated probabilities. This server wraps
``laya.Router`` so any MCP client (Claude Code, Claude Desktop, Cursor, ...) can
use it as a fast classifier, guardrail, triage or routing tool.

Configuration (environment variables):

  MCP_TRANSPORT   stdio | http (streamable HTTP) | sse        default: stdio
  MCP_HOST        bind address for http/sse                  default: 0.0.0.0
  MCP_PORT        bind port for http/sse                     default: 8000
  MCP_API_KEY     if set, http/sse require "Authorization: Bearer <key>"
  LAYA_DEVICE     torch device (cpu, cuda, mps)              default: auto
  LAYA_PRELOAD    load checkpoints at startup (1/0)          default: 0
  LAYA_MODELS     checkpoints to preload, comma separated    default: english,multilingual
  LAYA_MAX_LOADED how many checkpoints stay resident         default: 2
  LAYA_THREADS    cap torch CPU threads                      default: torch default
  LAYA_DEFAULT    checkpoint for undecided language          default: english
  LAYA_REQUIRE_GPU exit at startup if a preloaded checkpoint is not on CUDA (1/0)  default: 0
"""
import contextlib
import logging
import os
import sys
import threading
from typing import Any, Dict, List, Literal, Optional, Union

import anyio
from mcp.server.mcpserver import MCPServer

log = logging.getLogger("laya_mcp")

CHECKPOINTS = ("english", "multilingual", "typed-decisions")
PRESETS = ("triage", "email", "guard", "moderation", "router")

Model = Optional[Literal["english", "multilingual", "typed-decisions"]]
State = Union[str, Dict[str, Any], List[Any]]


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


class Engine:
    """Lazily builds a single shared ``laya.Router``.

    Heavy imports (torch, transformers) happen on first use so the MCP handshake is
    instant. Anything the model stack prints goes to stderr, because stdout carries
    the stdio JSON-RPC stream.
    """

    def __init__(self):
        self._router = None
        self._lock = threading.Lock()

    def router(self):
        with self._lock:
            if self._router is None:
                with contextlib.redirect_stdout(sys.stderr):
                    threads = os.environ.get("LAYA_THREADS")
                    if threads and threads.isdigit() and int(threads) > 0:
                        import torch

                        torch.set_num_threads(int(threads))
                    from laya import Router

                    self._router = Router(
                        device=os.environ.get("LAYA_DEVICE") or None,
                        max_loaded=int(os.environ.get("LAYA_MAX_LOADED", "2")),
                        default=os.environ.get("LAYA_DEFAULT", "english"),
                    )
            return self._router

    def preload(self):
        names = [m.strip() for m in os.environ.get("LAYA_MODELS", "english,multilingual").split(",") if m.strip()]
        log.info("preloading checkpoints: %s", names)
        router = self.router()
        with contextlib.redirect_stdout(sys.stderr):
            router.preload(names)
            # One throwaway forward per checkpoint so CUDA context/kernel setup is not paid
            # by the first real request.
            warm = {"w": {"type": "noul", "instructions": "Is `text` a greeting?"}}
            for name in names:
                router.predict({"text": "hello"}, warm, model=name)
        log.info("checkpoint devices: %s", self.devices())

    def devices(self) -> Dict[str, str]:
        """Where each loaded checkpoint actually runs. laya silently falls back to CPU when
        CUDA is unavailable or out of memory, so this is the ground truth, not LAYA_DEVICE."""
        agents = getattr(self._router, "_agents", None) or {}
        return {name: str(agent.device) for name, agent in list(agents.items())}

    def predict(self, state: State, questions: Dict[str, Any], model: Optional[str] = None,
                lang: Optional[str] = None) -> Dict[str, Any]:
        router = self.router()
        with contextlib.redirect_stdout(sys.stderr):
            return router.predict(state, questions, model=model, lang=lang)

    def loaded(self) -> List[str]:
        return [] if self._router is None else self._router.loaded


engine = Engine()

INSTRUCTIONS = """\
Laya is a fast (~30-400 ms), non-generative decision model. It never writes text: give it a
`state` (text or a JSON object) and typed questions, and it returns an answer plus calibrated
probabilities for each question in one forward pass. 100+ languages (auto-routed).

Question types:
  choice : pick one key from `criteria` (dict of key -> description). Best with <= 20 options.
  score  : ordinal scale, `criteria` is a list of level descriptions (low -> high); returns the
           expected level index as a float.
  noul   : yes/no probability (0..1). On English text prefer `laya_yes_no`, which asks the same
           question as a neutral two-option choice (more reliable).

Refer to state fields in instructions with backticks, e.g. "Is `body` a phishing attempt?".
Gate decisions on `confidence`.
Base checkpoints are over-confident zero-shot; treat probabilities as relative, not absolute.
"""

mcp = MCPServer("laya", instructions=INSTRUCTIONS)


async def _predict(state: State, questions: Dict[str, Any], model: Optional[str] = None,
                   lang: Optional[str] = None) -> Dict[str, Any]:
    if state is None or (isinstance(state, str) and not state.strip()):
        raise ValueError("state must be non-empty text or a JSON object")
    return await anyio.to_thread.run_sync(lambda: engine.predict(state, questions, model=model, lang=lang))


def _compact(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep the answer payload; drop the language detection block and `action`
    (act_probability reads ~1.0 for every input upstream and carries no signal)."""
    routing = result.get("routing") or {}
    answers = {qid: {k: v for k, v in a.items() if k != "action"} for qid, a in result.get("answers", {}).items()}
    return {
        "answers": answers,
        "routing": {"model": routing.get("model"), "reason": routing.get("reason")},
        "usage": result.get("usage"),
    }


@mcp.tool()
async def laya_decide(
    state: State,
    questions: Dict[str, Dict[str, Any]],
    model: Model = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Answer any number of typed questions about `state` in one forward pass.

    Args:
        state: Text, or a JSON object such as {"subject": "...", "body": "..."}.
        questions: Map of question id -> definition. Examples:
            {"department": {"type": "choice", "instructions": "Which team handles `body`?",
                            "criteria": {"billing": "invoices, refunds", "technical": "bugs", "other": "else"}},
             "urgency": {"type": "score", "instructions": "How urgent is it?",
                         "criteria": ["not urgent", "soon", "blocking"]},
             "refund": {"type": "noul", "instructions": "Does the sender ask for a refund?"}}
        model: Force a checkpoint (english | multilingual | typed-decisions). Omit to auto-route by language.
        lang: Optional ISO language code of the state (e.g. "tr", "en") to skip language detection.
    """
    return _compact(await _predict(state, questions, model=model, lang=lang))


@mcp.tool()
async def laya_classify(
    text: State,
    labels: Union[Dict[str, str], List[str]],
    instructions: str = "Which label best describes `text`?",
    model: Model = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify `text` into exactly one of `labels` with calibrated probabilities.

    Args:
        text: Text or JSON object to classify.
        labels: Either a list of label names, or a dict of label -> short description
                (descriptions noticeably improve accuracy). Keep it under ~20 labels.
        instructions: The classification question.
    """
    criteria = {str(l): None for l in labels} if isinstance(labels, list) else dict(labels)
    if len(criteria) < 2:
        raise ValueError("labels needs at least two entries")
    state = {"text": text} if isinstance(text, str) else text
    res = await _predict(state, {"label": {"type": "choice", "instructions": instructions, "criteria": criteria}},
                         model=model, lang=lang)
    ans = res["answers"]["label"]
    return {
        "label": ans["choice"],
        "confidence": ans["confidence"],
        "probabilities": ans["probabilities"],
        "routing": _compact(res)["routing"],
    }


@mcp.tool()
async def laya_yes_no(
    text: State,
    question: str,
    yes_means: Optional[str] = None,
    no_means: Optional[str] = None,
    model: Model = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Ask a yes/no question about `text`; returns P(yes).

    Implemented as a neutral two-option choice, the upstream-recommended workaround for the
    `noul` label bias on the English checkpoint.

    Args:
        text: Text or JSON object.
        question: e.g. "Is `text` a phishing attempt?"
        yes_means / no_means: Optional descriptions of each answer to sharpen the decision.
    """
    state = {"text": text} if isinstance(text, str) else text
    q = {
        "type": "choice",
        "instructions": question,
        "criteria": {"A": "yes" + (", " + yes_means if yes_means else ""),
                     "B": "no" + (", " + no_means if no_means else "")},
    }
    res = await _predict(state, {"q": q}, model=model, lang=lang)
    ans = res["answers"]["q"]
    p_yes = ans["probabilities"]["A"]
    return {
        "answer": "yes" if p_yes >= 0.5 else "no",
        "p_yes": p_yes,
        "confidence": ans["confidence"],
        "routing": _compact(res)["routing"],
    }


@mcp.tool()
async def laya_score(
    text: State,
    instructions: str,
    levels: List[str],
    model: Model = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Rate `text` on an ordinal scale.

    Args:
        text: Text or JSON object.
        instructions: e.g. "How frustrated does the customer sound?"
        levels: Level descriptions from lowest to highest, e.g. ["calm", "annoyed", "furious"].
    Returns the expected level (float, 0 = first level) and the per-level distribution.
    """
    if len(levels) < 2:
        raise ValueError("levels needs at least two entries")
    state = {"text": text} if isinstance(text, str) else text
    res = await _predict(state, {"s": {"type": "score", "instructions": instructions, "criteria": levels}},
                         model=model, lang=lang)
    ans = res["answers"]["s"]
    top = max(ans["probabilities"], key=ans["probabilities"].get)
    return {
        "score": ans["score"],
        "max_score": len(levels) - 1,
        "most_likely_level": levels[int(top)],
        "probabilities": {levels[int(k)]: v for k, v in ans["probabilities"].items()},
        "confidence": ans["confidence"],
        "routing": _compact(res)["routing"],
    }


def _preset_questions(name: str) -> tuple:
    """Return (state_field, questions) for a built-in preset."""
    import laya

    field, build = {
        "triage": ("message", laya.triage_questions),
        "email": ("body", laya.email_questions),
        "guard": ("prompt", laya.guard_questions),
        "moderation": ("post", laya.moderation_questions),
        "router": ("request", laya.router_questions),
    }[name]
    return field, build()


@mcp.tool()
async def laya_preset(
    preset: Literal["triage", "email", "guard", "moderation", "router"],
    text: str,
    subject: Optional[str] = None,
    sender: Optional[str] = None,
    model: Model = None,
    lang: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a ready-made question set on `text`.

    Presets:
      triage     - support ticket: intent, urgency, frustration, refund, churn risk
      email      - inbound email: team, spam, phishing, urgency, needs reply (uses subject/sender)
      guard      - LLM input guardrail: jailbreak, prompt injection, sensitive data, harm severity, topic
      moderation - user content: toxic, harassment, threat, spam, severity
      router     - LLM request routing: difficulty, domain, needs tools, sensitive
    """
    field, questions = await anyio.to_thread.run_sync(_preset_questions, preset)
    if preset == "email":
        from laya import email_state

        state = email_state(subject or "", text, sender=sender)
    else:
        state = {field: text}
    return _compact(await _predict(state, questions, model=model, lang=lang))


@mcp.tool()
async def laya_route(state: State, lang: Optional[str] = None) -> Dict[str, Any]:
    """Show which Laya checkpoint would handle `state` and why (language/script detection).

    Runs no model and loads no weights.
    """
    def _route():
        from laya import Router

        return dict(Router(default=os.environ.get("LAYA_DEFAULT", "english")).route(state, lang=lang))

    return await anyio.to_thread.run_sync(_route)


@mcp.tool()
async def laya_status() -> Dict[str, Any]:
    """Report which checkpoints are loaded in memory and the server configuration."""
    return {
        "loaded": engine.loaded(),
        "devices": engine.devices(),
        "available": list(CHECKPOINTS),
        "requested_device": os.environ.get("LAYA_DEVICE") or "auto",
        "max_loaded": int(os.environ.get("LAYA_MAX_LOADED", "2")),
        "presets": list(PRESETS),
    }


@mcp.resource("laya://presets/{name}")
def preset_resource(name: str) -> Dict[str, Any]:
    """The question definitions behind a preset, to copy and adapt for `laya_decide`."""
    if name not in PRESETS:
        raise ValueError("unknown preset %r; choose one of %s" % (name, ", ".join(PRESETS)))
    field, questions = _preset_questions(name)
    return {"state_field": field, "questions": questions}


class _BearerAuth:
    """Minimal ASGI middleware enforcing a static bearer token."""

    def __init__(self, app, key: str):
        self.app, self.expected = app, ("Bearer " + key).encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") != "/health":
            if dict(scope.get("headers") or []).get(b"authorization") != self.expected:
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json")]})
                await send({"type": "http.response.body", "body": b'{"error":"unauthorized"}'})
                return
        await self.app(scope, receive, send)


def _serve_http(transport: str) -> None:
    import uvicorn
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    host = os.environ.get("MCP_HOST", "0.0.0.0")
    port = int(os.environ.get("MCP_PORT", "8000"))
    # Passing the real bind host keeps the SDK's localhost-only DNS rebinding guard off when
    # serving on 0.0.0.0 (e.g. inside a container), and on when bound to localhost.
    app = mcp.streamable_http_app(host=host) if transport == "http" else mcp.sse_app(host=host)
    app.router.routes.append(Route("/health", lambda _req: JSONResponse(
        {"status": "ok", "loaded": engine.loaded(), "devices": engine.devices()})))
    key = os.environ.get("MCP_API_KEY")
    asgi = _BearerAuth(app, key) if key else app
    if not key:
        log.warning("MCP_API_KEY is not set: the HTTP endpoint is unauthenticated")
    uvicorn.run(asgi, host=host, port=port,
                log_level=os.environ.get("MCP_LOG_LEVEL", "info"))


def main() -> None:
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("MCP_LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # transformers probes TensorFlow at import; its runtime can deadlock model construction.
    os.environ.setdefault("USE_TF", "0")
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
    if len(sys.argv) > 1 and sys.argv[1] in ("stdio", "http", "sse"):
        transport = sys.argv[1]
    if _env_bool("LAYA_PRELOAD", False):
        engine.preload()
        if _env_bool("LAYA_REQUIRE_GPU", False):
            off_gpu = {n: d for n, d in engine.devices().items() if not d.startswith("cuda")}
            if off_gpu:
                raise SystemExit("LAYA_REQUIRE_GPU=1 but checkpoints are not on CUDA: %s. Check the "
                                 "NVIDIA driver, nvidia-container-toolkit and the compose GPU "
                                 "reservation, and use the :cuda image." % off_gpu)
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport in ("http", "sse"):
        _serve_http(transport)
    else:
        raise SystemExit("MCP_TRANSPORT must be stdio, http or sse, got %r" % transport)


if __name__ == "__main__":
    main()
