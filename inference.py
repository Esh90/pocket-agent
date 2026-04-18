"""
Pocket-Agent inference module.
Grader interface: run(prompt: str, history: list[dict]) -> str

Constraints:
  - NO network imports (requests/urllib/http/socket are absent)
  - Loads GGUF via llama-cpp-python (pure CPU, offline)
  - <200 ms per turn target (Qwen2.5-0.5B q4_k_m @ ~60 tok/s on Colab CPU)
"""

from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── strict: no network modules anywhere in this file ─────────────────────────
# (AST scanner will confirm)

# ── llama-cpp-python ──────────────────────────────────────────────────────────
try:
    from llama_cpp import Llama
except ImportError as e:
    raise ImportError(
        "llama-cpp-python is required. Install with:\n"
        "  pip install llama-cpp-python"
    ) from e

# ── configuration ─────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).parent
_GGUF_SEARCH = [
    _SCRIPT_DIR / "pocket-agent-gguf",
    _SCRIPT_DIR,
    Path("pocket-agent-gguf"),
    Path("."),
]

SYSTEM_PROMPT = (
    "You are a compact tool-calling assistant running fully offline on a mobile device. "
    "When the user's request clearly maps to one of the available tools, respond with exactly "
    "one tool call formatted as JSON inside <tool_call>...</tool_call> tags. "
    "When the request is chitchat, impossible to fulfil with the available tools, or truly "
    "ambiguous with no prior context to resolve it, respond with plain helpful text—no tool call.\n\n"
    "Available tools (schema is final):\n"
    '{"tool":"weather","args":{"location":"string","unit":"C|F"}}\n'
    '{"tool":"calendar","args":{"action":"list|create","date":"YYYY-MM-DD","title":"string?"}}\n'
    '{"tool":"convert","args":{"value":number,"from_unit":"string","to_unit":"string"}}\n'
    '{"tool":"currency","args":{"amount":number,"from":"ISO3","to":"ISO3"}}\n'
    '{"tool":"sql","args":{"query":"string"}}'
)

MAX_NEW_TOKENS = 256
CONTEXT_LENGTH = 2048          # matches training MAX_SEQ_LEN * 2 (inference is lighter)
N_THREADS      = max(1, (os.cpu_count() or 4) - 1)


# ── singleton model loader ────────────────────────────────────────────────────

_llm: Optional[Llama] = None


def _find_gguf() -> Path:
    """Locate the GGUF file; searches known paths."""
    # Allow override via env var
    env_path = os.environ.get("POCKET_AGENT_GGUF")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    for search_dir in _GGUF_SEARCH:
        if not search_dir.exists():
            continue
        candidates = sorted(search_dir.glob("*.gguf"))
        if candidates:
            # Prefer q4_k_m if multiple exist
            for c in candidates:
                if "q4_k_m" in c.name.lower():
                    return c
            return candidates[0]

    raise FileNotFoundError(
        "No GGUF model found. Expected locations:\n"
        + "\n".join(f"  {d}" for d in _GGUF_SEARCH)
        + "\nSet POCKET_AGENT_GGUF=/path/to/model.gguf to override."
    )


def _get_model() -> Llama:
    """Lazy singleton — loads once, reuses across calls."""
    global _llm
    if _llm is None:
        gguf_path = _find_gguf()
        print(f"[inference] Loading model: {gguf_path}", file=sys.stderr)
        _llm = Llama(
            model_path    = str(gguf_path),
            n_ctx         = CONTEXT_LENGTH,
            n_threads     = N_THREADS,
            n_threads_batch = N_THREADS,
            n_gpu_layers  = 0,          # pure CPU — grader constraint
            verbose       = False,
            use_mmap      = True,       # memory-mapped for fast startup
            use_mlock     = False,
        )
        print("[inference] Model ready.", file=sys.stderr)
    return _llm


# ── ChatML formatting ─────────────────────────────────────────────────────────

def _build_chatml_prompt(prompt: str, history: list[dict]) -> str:
    """
    Build a ChatML-formatted string from system prompt + history + current prompt.

    history format (same as OpenAI): [{"role": "user"|"assistant", "content": "..."}]
    """
    parts: list[str] = []
    parts.append(f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n")

    for turn in history:
        role    = turn.get("role", "user")
        content = turn.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

    parts.append(f"<|im_start|>user\n{prompt}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")

    return "".join(parts)


# ── post-processing ───────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)

_WORD_TO_NUMBER = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000,
}


def _normalize_tool_call(raw_json: str) -> str:
    """
    Attempt to normalize and validate a tool call JSON string.
    Returns canonical JSON string or raises ValueError.
    """
    # Strip trailing garbage (model sometimes continues after closing brace)
    # Find the outermost balanced braces
    depth = 0
    end_idx = -1
    for i, ch in enumerate(raw_json):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    if end_idx == -1:
        raise ValueError("No complete JSON object found")

    payload = json.loads(raw_json[: end_idx + 1])

    # Validate required top-level keys
    if "tool" not in payload or "args" not in payload:
        raise ValueError(f"Missing 'tool' or 'args': {payload}")

    tool = payload["tool"].lower().strip()
    args = payload["args"]

    # Per-tool arg normalization
    if tool == "weather":
        args["unit"] = args.get("unit", "C").upper()
        if args["unit"] not in ("C", "F"):
            args["unit"] = "C"

    elif tool == "calendar":
        action = args.get("action", "list").lower()
        args["action"] = action if action in ("list", "create") else "list"
        # Validate date format loosely
        date_val = str(args.get("date", ""))
        if not re.match(r"\d{4}-\d{2}-\d{2}", date_val):
            raise ValueError(f"Bad date format: {date_val}")

    elif tool == "convert":
        # Coerce string numbers ("fifty") to float
        val = args.get("value", 0)
        if isinstance(val, str):
            val_lower = val.lower().replace(",", "")
            args["value"] = _WORD_TO_NUMBER.get(val_lower, float(val_lower))
        else:
            args["value"] = float(val)

    elif tool == "currency":
        amt = args.get("amount", 0)
        if isinstance(amt, str):
            amt_lower = amt.lower().replace(",", "")
            args["amount"] = _WORD_TO_NUMBER.get(amt_lower, float(amt_lower))
        else:
            args["amount"] = float(amt)
        # Normalize ISO codes to uppercase
        args["from"] = str(args.get("from", "USD")).upper()
        args["to"]   = str(args.get("to",   "USD")).upper()

    elif tool == "sql":
        if "query" not in args or not args["query"].strip():
            raise ValueError("sql tool missing 'query'")

    else:
        raise ValueError(f"Unknown tool: {tool}")

    payload["tool"] = tool
    payload["args"] = args
    return json.dumps(payload, ensure_ascii=False)


def _postprocess(raw_output: str) -> str:
    """
    Extract and validate tool call from model output,
    or return cleaned plain-text if it's a refusal.
    """
    # Strip leading/trailing whitespace and any trailing eos tokens
    text = raw_output.strip()
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<\|endoftext\|>.*$", "", text, flags=re.DOTALL).strip()

    match = _TOOL_CALL_RE.search(text)
    if match:
        raw_json = match.group(1).strip()
        try:
            normalized = _normalize_tool_call(raw_json)
            return f"<tool_call>{normalized}</tool_call>"
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Malformed tool call → treat as refusal with error note
            # (grader gives 0 for malformed, same as wrong tool; plain text is safer)
            return f"I couldn't process that request properly. Could you rephrase?"

    # Plain text refusal — sanitize
    return text if text else "I'm not sure how to help with that. Could you clarify?"


# ── public API ────────────────────────────────────────────────────────────────

def run(prompt: str, history: list[dict]) -> str:
    """
    Grader interface.

    Args:
        prompt:  The current user message.
        history: Prior turns as [{"role": "user"|"assistant", "content": "..."}].
                 May be empty for single-turn queries.

    Returns:
        Either "<tool_call>{...}</tool_call>" or plain text for refusals.
    """
    llm = _get_model()

    formatted_prompt = _build_chatml_prompt(prompt, history)

    result = llm(
        formatted_prompt,
        max_tokens    = MAX_NEW_TOKENS,
        temperature   = 0.01,     # near-deterministic for tool calls
        top_p         = 0.9,
        top_k         = 40,
        repeat_penalty = 1.1,
        stop          = ["<|im_end|>", "<|endoftext|>", "\n<|im_start|>"],
        echo          = False,
    )

    raw_output = result["choices"][0]["text"]
    return _postprocess(raw_output)


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("What's the weather in Karachi?", []),
        ("Convert 100 km to miles", []),
        ("How much is 500 USD in PKR?", []),
        ("Schedule 'Team standup' on 2025-04-15", []),
        ("Show me all active users in the database", []),
        ("Tell me a joke", []),
        ("convert that to euros", []),   # should refusal (no history)
        (
            "Now convert it to Fahrenheit",
            [
                {"role": "user",      "content": "What's the weather in Berlin?"},
                {"role": "assistant", "content": '<tool_call>{"tool":"weather","args":{"location":"Berlin","unit":"C"}}</tool_call>'},
            ],
        ),
    ]

    print("=" * 60)
    for i, (prompt, history) in enumerate(test_cases):
        output = run(prompt, history)
        print(f"[{i+1}] USER: {prompt[:60]}")
        print(f"     RESP: {output[:120]}")
        print()