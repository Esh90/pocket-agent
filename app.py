"""
Pocket-Agent — Gradio chatbot demo.
Tested on Gradio 6.x.
"""

from __future__ import annotations
import json
import re

import gradio as gr
from inference import run

_TOOL_ICONS = {"weather":"🌤️","calendar":"📅","convert":"🔄","currency":"💱","sql":"🗄️"}
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE)


def _format_response(raw: str) -> str:
    match = _TOOL_CALL_RE.search(raw)
    if not match:
        return raw
    try:
        payload = json.loads(match.group(1))
        tool  = payload.get("tool", "unknown")
        args  = payload.get("args", {})
        icon  = _TOOL_ICONS.get(tool, "🔧")
        lines = "\n".join(f"  **{k}**: `{v}`" for k, v in args.items())
        return f"{icon} **Tool call: `{tool}`**\n\n{lines}\n\n---\n*Raw:* `{match.group(0)}`"
    except (json.JSONDecodeError, KeyError):
        return raw


def _recover_raw(display_msg: str) -> str:
    """Strip display formatting to get back the raw tool call string."""
    m = re.search(r"`(<tool_call>.*?</tool_call>)`", str(display_msg), re.DOTALL)
    return m.group(1) if m else str(display_msg)


def chat(user_message: str, history: list[dict]):
    """
    Gradio 6.x chat function.
    history is a list of {"role": "user"|"assistant", "content": "..."} dicts.
    """
    if not user_message.strip():
        return "", history

    # Build clean API history (undo display formatting on assistant messages)
    api_history = []
    for msg in history:
        role    = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            api_history.append({"role": "user", "content": content})
        elif role == "assistant":
            api_history.append({"role": "assistant", "content": _recover_raw(content)})

    raw_response     = run(user_message.strip(), api_history)
    display_response = _format_response(raw_response)

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": display_response},
    ]
    return "", history


EXAMPLES = [
    ["What's the weather in Karachi?"],
    ["Convert 100 km to miles"],
    ["How much is 500 USD in PKR?"],
    ["Schedule 'Team standup' on 2025-04-20"],
    ["Show me all active users"],
    ["Tell me a joke"],
    ["aaj Lahore mein mausam kaisa hai"],
    ["weatehr in Dubai Fahrenheit mein"],
]

with gr.Blocks(title="Pocket-Agent Demo") as demo:
    gr.Markdown("""
# 🤖 Pocket-Agent — On-Device Tool-Calling Assistant
**Model:** Qwen2.5-0.5B-Instruct (q4_k_m GGUF, ~220MB) | **Mode:** Fully offline CPU inference

Ask me to check weather, manage your calendar, convert units/currencies, or run SQL queries.
    """)

    chatbot = gr.Chatbot(
        height=420,
        show_label=False,
        # Gradio 6.12 no longer supports the `type` argument.
        render_markdown=True,
    )

    with gr.Row():
        msg_box  = gr.Textbox(
            placeholder="e.g. 'weather in Tokyo' or 'convert 50 kg to lbs'",
            show_label=False,
            scale=9,
            container=False,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("🗑️ Clear conversation", size="sm")

    gr.Examples(examples=EXAMPLES, inputs=msg_box, label="Example queries (click to load)")

    gr.Markdown("""
---
**Tool output format:** `<tool_call>{"tool": "...", "args": {...}}</tool_call>`  
**Refusals:** Plain text when no tool matches (chitchat, unknown tools, ambiguous references).
    """)

    send_btn.click(chat, [msg_box, chatbot], [msg_box, chatbot])
    msg_box.submit(chat, [msg_box, chatbot], [msg_box, chatbot])
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = True,
        show_error  = True,
        inbrowser   = False,
    )