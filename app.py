import json
import os
import re
import sys
import time
import html
from typing import Dict, Optional

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

# Ensure project root is in sys.path.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agent.agent import ReActAgent
from src.agent.agent_v2 import ReActAgentV2
from src.chatbot.chatbot import ChatbotBaseline
from src.core.llm_provider import LLMProvider
from src.tools.tools import TOOL_SPECS

load_dotenv()

st.set_page_config(
    page_title="Lab 03: Chatbot vs ReAct Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    :root {
        --paper: #EEF6FB;
        --surface: #F7FBFF;
        --surface-2: #EDF5FA;
        --ink: #10202B;
        --muted: #5F7482;
        --line: #D7E6F0;
        --accent: #0B3A5B;
        --accent-2: #0E5E8F;
        --accent-soft: #DCECF6;
        --shadow-dark: rgba(122, 151, 169, 0.34);
        --shadow-light: rgba(255, 255, 255, 0.95);
    }
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"],
    [data-testid="stDeployButton"],
    #MainMenu,
    footer {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }
    html, body, [data-testid="stAppViewContainer"], .main {
        background: var(--paper);
    }
    .block-container {
        padding-top: 0.65rem;
        padding-bottom: 0.45rem;
        max-width: 1480px;
    }
    [data-testid="stSidebar"] {
        display: none;
    }
    .screen-title {
        font-size: 1.1rem;
        font-weight: 750;
        color: var(--ink);
        margin: 0 0 0.25rem 0;
        letter-spacing: 0;
    }
    .eyebrow {
        color: var(--accent);
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.72rem;
        font-weight: 750;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }
    .panel {
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 8px;
        background: var(--surface);
        padding: 0.75rem;
        height: calc(100vh - 314px);
        min-height: 330px;
        overflow-y: auto;
        box-shadow: 10px 10px 24px var(--shadow-dark), -10px -10px 24px var(--shadow-light);
    }
    .control-panel {
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 8px;
        background: var(--surface);
        padding: 0.75rem;
        margin-bottom: 0.65rem;
        box-shadow: 10px 10px 24px var(--shadow-dark), -10px -10px 24px var(--shadow-light);
    }
    .panel-title {
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.78rem;
        font-weight: 750;
        color: var(--accent);
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }
    .muted {
        color: var(--muted);
        font-size: 0.78rem;
    }
    .chat-shell {
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 8px;
        background: var(--surface);
        padding: 0.7rem 0.85rem;
        height: calc(100vh - 255px);
        min-height: 420px;
        overflow-y: auto;
        box-shadow: inset 7px 7px 18px rgba(122, 151, 169, 0.18), inset -7px -7px 18px rgba(255,255,255,0.9);
    }
    .message-row {
        display: flex;
        width: 100%;
        margin: 0.55rem 0;
    }
    .message-row.user {
        justify-content: flex-end;
    }
    .message-row.assistant {
        justify-content: flex-start;
    }
    .message-bubble {
        max-width: 72%;
        border-radius: 8px;
        padding: 0.65rem 0.8rem;
        color: var(--ink);
        line-height: 1.48;
        word-break: break-word;
        box-shadow: 6px 6px 14px rgba(122, 151, 169, 0.22), -6px -6px 14px rgba(255,255,255,0.95);
    }
    .message-bubble.user {
        background: var(--accent);
        color: #FFFFFF;
        border-bottom-right-radius: 3px;
    }
    .message-bubble.assistant {
        background: var(--surface-2);
        border-bottom-left-radius: 3px;
    }
    .message-bubble strong {
        font-weight: 800;
    }
    .message-bubble code {
        border-radius: 5px;
        padding: 0.08rem 0.28rem;
        background: rgba(11, 58, 91, 0.10);
        color: var(--accent);
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        font-size: 0.88em;
    }
    .message-bubble.user code {
        background: rgba(255, 255, 255, 0.18);
        color: #FFFFFF;
    }
    .message-label {
        display: block;
        font-size: 0.68rem;
        font-weight: 750;
        opacity: 0.78;
        margin-bottom: 0.22rem;
        text-transform: uppercase;
    }
    .mode-switch-title {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 700;
        margin: 0.2rem 0 0.35rem 0;
    }
    .prompt-panel {
        border: 1px solid rgba(255, 255, 255, 0.72);
        border-radius: 8px;
        background: var(--surface);
        padding: 0.45rem 0.65rem 0.2rem 0.65rem;
        margin-top: 0.55rem;
        box-shadow: 8px 8px 18px var(--shadow-dark), -8px -8px 18px var(--shadow-light);
    }
    .sample-row {
        margin: 0.35rem 0 0.25rem 0;
    }
    .tool-chip {
        display: inline-block;
        background: var(--accent-soft);
        border: 1px solid #BDD4E3;
        color: var(--accent);
        border-radius: 6px;
        padding: 4px 8px;
        margin: 0 6px 6px 0;
        font-size: 0.82rem;
        font-weight: 650;
    }
    div[data-testid="stMetric"] {
        background: var(--surface-2);
        border: 1px solid rgba(255, 255, 255, 0.68);
        border-radius: 8px;
        padding: 0.35rem 0.45rem;
        box-shadow: inset 4px 4px 9px rgba(122, 151, 169, 0.18), inset -4px -4px 9px rgba(255,255,255,0.9);
    }
    div[data-testid="stMetricValue"] {
        color: var(--ink);
        font-size: 0.92rem;
    }
    div[data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.68rem;
    }
    .stChatMessage {
        border-radius: 8px;
    }
    .stButton button {
        border-radius: 8px;
        border-color: rgba(255, 255, 255, 0.72);
        color: var(--accent);
        background: var(--surface);
        min-height: 2.05rem;
        padding: 0.25rem 0.55rem;
        box-shadow: 5px 5px 12px rgba(122, 151, 169, 0.24), -5px -5px 12px rgba(255,255,255,0.95);
    }
    .stButton button[kind="primary"] {
        background: var(--accent);
        border-color: var(--accent);
        color: #FFFFFF;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: #FFFFFF !important;
        border-radius: 8px;
        min-height: 2.15rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.15rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2rem;
        padding: 0 0.55rem;
        color: var(--muted);
    }
    [data-testid="stRadio"] [role="radiogroup"] {
        gap: 0.35rem;
    }
    [data-testid="stRadio"] [role="radiogroup"] label {
        min-height: 2.15rem;
        border-radius: 999px;
        padding: 0.25rem 0.65rem;
        border: 1px solid rgba(255,255,255,0.72);
        box-shadow: 4px 4px 10px rgba(122, 151, 169, 0.20), -4px -4px 10px rgba(255,255,255,0.94);
        background: var(--surface);
        font-weight: 750;
    }
    [data-testid="stRadio"] [role="radiogroup"] label * {
        font-weight: 750 !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] label:nth-child(1) {
        color: #0B3A5B;
        background: linear-gradient(135deg, #DCECF6, #F7FBFF);
    }
    [data-testid="stRadio"] [role="radiogroup"] label:nth-child(2) {
        color: #236B5B;
        background: linear-gradient(135deg, #DDF3EC, #F7FBFF);
    }
    [data-testid="stRadio"] [role="radiogroup"] label:nth-child(3) {
        color: #634E9A;
        background: linear-gradient(135deg, #E8E2FA, #F7FBFF);
    }
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        color: #FFFFFF !important;
        box-shadow: inset 4px 4px 9px rgba(33, 63, 82, 0.32), inset -4px -4px 9px rgba(255,255,255,0.22);
    }
    [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) * {
        color: #FFFFFF !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] label:not(:has(input:checked)) * {
        color: inherit !important;
    }
    [data-testid="stRadio"] [role="radiogroup"] label:nth-child(1):has(input:checked) {
        background: linear-gradient(135deg, #1C5F8A, #2C8DBD);
    }
    [data-testid="stRadio"] [role="radiogroup"] label:nth-child(2):has(input:checked) {
        background: linear-gradient(135deg, #23816B, #48B997);
    }
    [data-testid="stRadio"] [role="radiogroup"] label:nth-child(3):has(input:checked) {
        background: linear-gradient(135deg, #6858B5, #9184E8);
    }
    div[data-testid="stExpander"] {
        background: var(--surface-2);
        border-color: var(--line);
        border-radius: 8px;
    }
    .trace-scroll {
        max-height: calc(100vh - 430px);
        overflow-y: auto;
        padding-right: 0.2rem;
    }
    .block-container > div {
        gap: 0.75rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


DEFAULT_PROMPT = (
    "Tôi muốn mua 2 iPhone, dùng mã 'WINNER' và giao tới Hà Nội. "
    "Khối lượng 0.8 kg. Tổng bao nhiêu?"
)

GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]
OPENAI_MODELS = ["gpt-4o-mini", "text-embedding-3-small"]
GEMINI_DEFAULT_MODEL = GEMINI_MODELS[0]
OPENAI_DEFAULT_MODEL = OPENAI_MODELS[0]

SAMPLE_PROMPTS = {
    "2 iPhone + WINNER": DEFAULT_PROMPT,
    "MacBook + Saigon": "Can I buy 1 MacBook and ship to Saigon? How much?",
    "iPad + LEGACY": (
        "I want to buy 1 iPad using code 'LEGACY' and ship to Saigon. "
        "The package weight is 0.5 kg. How much?"
    ),
    "Return policy": "What is your return policy?",
}


class DemoScriptedLLM(LLMProvider):
    """Deterministic provider for offline UI demos."""

    def __init__(self):
        super().__init__(model_name="demo-scripted-llm")

    def generate(self, prompt: str, system_prompt: Optional[str] = None):
        p_lower = prompt.lower()
        observation_count = p_lower.count("observation:")

        if "iphone" in p_lower and ("winner" in p_lower or "2 iphone" in p_lower):
            if observation_count == 0:
                content = (
                    "Thought: Need stock and price evidence for iPhone.\n"
                    'Action: check_stock({"item_name": "iPhone"})'
                )
            elif observation_count == 1:
                content = (
                    "Thought: Need to validate coupon WINNER.\n"
                    'Action: get_discount({"coupon_code": "WINNER"})'
                )
            elif observation_count == 2:
                content = (
                    "Thought: Need shipping fee for 0.8kg to Hanoi.\n"
                    'Action: calc_shipping({"weight": 0.8, "destination": "Hanoi"})'
                )
            else:
                content = (
                    "Thought: All evidence is available from tools.\n"
                    "Final Answer: Tổng chi phí mua 2 iPhone sau giảm giá WINNER 10% "
                    "và phí ship Hà Nội 38.000đ là 45.038.000 VNĐ."
                )
        elif "macbook" in p_lower:
            if observation_count == 0:
                content = (
                    "Thought: Need stock evidence for MacBook.\n"
                    'Action: check_stock({"item_name": "MacBook"})'
                )
            else:
                content = (
                    "Thought: MacBook stock is 0, so the order cannot proceed.\n"
                    "Final Answer: Rất tiếc, MacBook hiện đang hết hàng nên chưa thể "
                    "tạo đơn giao tới Saigon."
                )
        elif "ipad" in p_lower or "legacy" in p_lower:
            if observation_count == 0:
                content = (
                    "Thought: Need stock and price evidence for iPad.\n"
                    'Action: check_stock({"item_name": "iPad"})'
                )
            elif observation_count == 1:
                content = (
                    "Thought: Need to validate coupon LEGACY.\n"
                    'Action: get_discount({"coupon_code": "LEGACY"})'
                )
            elif observation_count == 2:
                content = (
                    "Thought: LEGACY is invalid, but shipping still needs calculation.\n"
                    'Action: calc_shipping({"weight": 0.5, "destination": "Saigon"})'
                )
            else:
                content = (
                    "Thought: All required evidence is available.\n"
                    'Final Answer: Mã "LEGACY" đã hết hạn. Tổng giá 1 iPad giao '
                    "Saigon gồm phí ship 47.500đ là 18.047.500 VNĐ."
                )
        elif "return policy" in p_lower or "đổi trả" in p_lower:
            content = (
                "Thought: This is a policy question and does not require tools.\n"
                "Final Answer: Cửa hàng hỗ trợ đổi trả trong vòng 30 ngày cho sản phẩm "
                "còn nguyên tem mác và đáp ứng điều kiện bảo hành."
            )
        elif "working hours" in p_lower or "giờ làm việc" in p_lower:
            content = (
                "Thought: This is a static store information question.\n"
                "Final Answer: Cửa hàng mở cửa từ 8:00 đến 22:00 hằng ngày."
            )
        else:
            if observation_count == 0:
                content = (
                    "Thought: Need to inspect stock before answering a commerce request.\n"
                    'Action: check_stock({"item_name": "iPhone"})'
                )
            else:
                content = (
                    "Thought: I have a tool observation and can answer concisely.\n"
                    "Final Answer: Cửa hàng đã kiểm tra dữ liệu liên quan và sẵn sàng hỗ trợ yêu cầu của bạn."
                )

        return {
            "content": content,
            "usage": {"prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
            "latency_ms": 120.0,
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None):
        yield "Demo stream"


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chào bạn. Nhập yêu cầu mua hàng hoặc chọn prompt mẫu để bắt đầu.",
            }
        ]
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    if "chat_draft" not in st.session_state:
        st.session_state.chat_draft = ""
    if "chat_input_version" not in st.session_state:
        st.session_state.chat_input_version = 0
    if "pending_request" not in st.session_state:
        st.session_state.pending_request = None


def set_chat_draft(prompt: str):
    st.session_state.chat_draft = prompt
    st.session_state.chat_input_version += 1


def clear_chat_draft():
    st.session_state.chat_draft = ""
    st.session_state.chat_input_version += 1


def ensure_total_tokens(usage: Optional[Dict[str, int]]) -> Dict[str, int]:
    usage = usage or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def normalize_tool_calls(tool_calls, trajectory):
    normalized = []
    for call in tool_calls or []:
        normalized.append(
            {
                "tool": call.get("tool", "unknown"),
                "args": call.get("args", {}),
                "result": call.get("result"),
            }
        )

    for item in trajectory or []:
        if item.get("type") != "action_and_observation":
            continue
        already_present = any(
            call["tool"] == item.get("tool") and call["args"] == item.get("args", {})
            for call in normalized
        )
        if already_present:
            for call in normalized:
                if call["tool"] == item.get("tool") and call["args"] == item.get("args", {}) and call.get("result") is None:
                    call["result"] = item.get("observation", {})
                    break
        else:
            normalized.append(
                {
                    "tool": item.get("tool", "unknown"),
                    "args": item.get("args", {}),
                    "result": item.get("observation", {}),
                }
            )
    return normalized


def compact_json(data) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))


def extract_thought(text: str) -> str:
    if not text:
        return ""
    thought = text.split("Action:", 1)[0].split("Final Answer:", 1)[0]
    return thought.replace("Thought:", "").strip()


def render_message_text(text: str) -> str:
    safe_text = html.escape(text or "")
    safe_text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe_text)
    safe_text = re.sub(r"`([^`]+)`", r"<code>\1</code>", safe_text)
    return safe_text.replace("\n", "<br>")


def build_trace_lines(trajectory) -> list[str]:
    lines = []
    for item in trajectory or []:
        step = item.get("step", "?")
        item_type = item.get("type", "log")
        if item_type == "llm_output":
            thought = extract_thought(item.get("content", ""))
            if thought:
                lines.append(f"Bước {step}: Agent suy nghĩ {thought}")
        elif item_type == "action_and_observation":
            tool_name = item.get("tool", "unknown")
            tool_args = compact_json(item.get("args", {}))
            observation = compact_json(item.get("observation", {}))
            lines.append(f"Bước {step}: Agent gọi tool {tool_name} với tham số {tool_args}.")
            lines.append(f"Observation nhận được: {observation}")
        elif item_type == "guardrail_trigger":
            lines.append(f"Bước {step}: Guardrail {item.get('guardrail', 'unknown')} được kích hoạt. {item.get('message', '')}".strip())
        elif item_type == "error":
            lines.append(f"Lỗi xử lý: {item.get('message', '')}")
    return lines


def build_tool_lines(tool_calls) -> list[str]:
    if not tool_calls:
        return ["Không có tool call trong lượt chạy này."]
    lines = []
    for idx, call in enumerate(tool_calls, start=1):
        lines.append(f"{idx}. Agent gọi {call.get('tool', 'unknown')} với tham số {compact_json(call.get('args', {}))}.")
        if call.get("result") is not None:
            lines.append(f"Observation trả về: {compact_json(call.get('result'))}")
    return lines


def provider_from_config(engine_type: str, openai_model: str, gemini_model: str) -> LLMProvider:
    if engine_type == "Offline":
        return DemoScriptedLLM()

    if engine_type == "Gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Chưa cấu hình GEMINI_API_KEY trong file .env.")
        from src.core.gemini_provider import GeminiProvider

        return GeminiProvider(api_key=api_key, model_name=gemini_model.strip() or GEMINI_DEFAULT_MODEL)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Chưa cấu hình OPENAI_API_KEY trong file .env.")
    if openai_model == "text-embedding-3-small":
        raise RuntimeError("text-embedding-3-small là embedding model, không dùng được cho chat completion hiện tại.")
    from src.core.openai_provider import OpenAIProvider

    return OpenAIProvider(api_key=api_key, model_name=openai_model.strip() or OPENAI_DEFAULT_MODEL)


def build_run_summary(run_info: Dict) -> str:
    tool_calls = run_info.get("tool_calls", [])
    if tool_calls:
        tools = ", ".join(call.get("tool", "unknown") for call in tool_calls[:3])
        if len(tool_calls) > 3:
            tools += f", +{len(tool_calls) - 3}"
    else:
        tools = "none"
    return (
        f"Status: {run_info.get('status', 'unknown')} | "
        f"Thinking: {run_info.get('thinking_count', 0)} | "
        f"Tools: {tools} | "
        f"Latency: {run_info.get('latency_ms', 0):.0f}ms"
    )


def run_prompt(prompt: str, mode: str, engine_type: str, openai_model: str, gemini_model: str, max_steps: int):
    provider = provider_from_config(engine_type, openai_model, gemini_model)
    started = time.time()

    if mode == "Chatbot":
        bot = ChatbotBaseline(provider)
        result = bot.chat(prompt)
        usage = ensure_total_tokens(result.get("usage"))
        assistant_text = result.get("response", "")
        run_info = {
            "kind": "chatbot",
            "mode": "Chatbot Baseline",
            "engine": engine_type,
            "model": provider.model_name,
            "status": "success",
            "final_answer": assistant_text,
            "latency_ms": result.get("latency_ms", 0),
            "usage": usage,
            "steps": 1,
            "thinking_count": 0,
            "log_events": 1,
            "tool_calls": [],
            "trajectory": [],
        }
    else:
        agent_cls = ReActAgentV2 if mode == "ReAct V2" else ReActAgent
        agent = agent_cls(provider, max_steps=max_steps)
        result = agent.run(prompt)
        usage = ensure_total_tokens(result.get("usage"))
        trajectory = result.get("trajectory", [])
        tool_calls = normalize_tool_calls(result.get("tool_call_sequence", []), trajectory)
        assistant_text = result.get("final_answer", "")
        run_info = {
            "kind": "agent",
            "mode": mode,
            "engine": engine_type,
            "model": provider.model_name,
            "status": result.get("status", "unknown"),
            "final_answer": assistant_text,
            "latency_ms": result.get("latency_ms", (time.time() - started) * 1000),
            "usage": usage,
            "steps": result.get("steps", 0),
            "thinking_count": sum(1 for item in trajectory if item.get("type") == "llm_output"),
            "log_events": len(trajectory),
            "tool_calls": tool_calls,
            "trajectory": trajectory,
        }

    return {
        "role": "assistant",
        "content": assistant_text,
        "run": run_info,
        "summary": build_run_summary(run_info),
    }


def enqueue_prompt(prompt: str, mode: str, engine_type: str, openai_model: str, gemini_model: str, max_steps: int):
    request_id = f"req-{time.time_ns()}"
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Đang suy nghĩ...",
            "status": "pending",
            "summary": f"{mode} đang chuẩn bị xử lý prompt.",
            "request_id": request_id,
        }
    )
    st.session_state.pending_request = {
        "id": request_id,
        "prompt": prompt,
        "mode": mode,
        "engine_type": engine_type,
        "openai_model": openai_model,
        "gemini_model": gemini_model,
        "max_steps": max_steps,
    }
    clear_chat_draft()


def complete_pending_request():
    pending = st.session_state.get("pending_request")
    if not pending:
        return False

    try:
        assistant_message = run_prompt(
            pending["prompt"],
            pending["mode"],
            pending["engine_type"],
            pending["openai_model"],
            pending["gemini_model"],
            pending["max_steps"],
        )
    except Exception as exc:
        run_info = {
            "kind": "error",
            "mode": pending["mode"],
            "engine": pending["engine_type"],
            "model": pending["gemini_model"] if pending["engine_type"] == "Gemini" else pending["openai_model"],
            "status": "error",
            "final_answer": str(exc),
            "latency_ms": 0,
            "usage": ensure_total_tokens({}),
            "steps": 0,
            "thinking_count": 0,
            "log_events": 1,
            "tool_calls": [],
            "trajectory": [{"step": 0, "type": "error", "message": str(exc)}],
        }
        assistant_message = {
            "role": "assistant",
            "content": "Mình gặp lỗi khi xử lý prompt. Xem tóm tắt lỗi bên dưới.",
            "run": run_info,
            "summary": f"Error: {str(exc)[:140]}",
        }

    for idx, message in enumerate(st.session_state.messages):
        if message.get("request_id") == pending["id"]:
            st.session_state.messages[idx] = assistant_message
            break

    st.session_state.last_run = assistant_message.get("run")
    st.session_state.pending_request = None
    return True


def get_latest_run():
    if st.session_state.get("last_run"):
        return st.session_state.last_run
    for message in reversed(st.session_state.get("messages", [])):
        if message.get("role") == "assistant" and message.get("run"):
            st.session_state.last_run = message["run"]
            return message["run"]
    return None


def render_control_panel():
    st.markdown('<div class="panel-title">Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="mode-switch-title">Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Mode",
        ["ReAct V2", "ReAct V1", "Chatbot"],
        horizontal=True,
        index=0,
        key="mode_switch",
        label_visibility="collapsed",
    )

    engine_type = st.radio(
        "Provider",
        ["Offline", "Gemini", "OpenAI"],
        horizontal=True,
        index=0,
        key="engine_switch",
    )

    if engine_type == "OpenAI":
        model_name = st.selectbox("Model", OPENAI_MODELS, index=0, key="openai_model")
        if model_name == "text-embedding-3-small":
            st.caption("Lưu ý: model embedding này không tương thích với Chat Completions hiện tại.")
        gemini_model = st.session_state.get("gemini_model", GEMINI_DEFAULT_MODEL)
    elif engine_type == "Gemini":
        gemini_model = st.selectbox("Model", GEMINI_MODELS, index=0, key="gemini_model")
        model_name = st.session_state.get("openai_model", OPENAI_DEFAULT_MODEL)
    else:
        st.text_input("Model", value="demo-scripted-llm", disabled=True)
        model_name = st.session_state.get("openai_model", OPENAI_DEFAULT_MODEL)
        gemini_model = st.session_state.get("gemini_model", GEMINI_DEFAULT_MODEL)

    col_steps, col_reset = st.columns([0.62, 0.38])
    with col_steps:
        max_steps = st.slider("Max steps", 1, 10, 6, key="max_steps")

    with col_reset:
        st.write("")
        if st.button("Clear", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Chào bạn. Nhập yêu cầu mua hàng hoặc chọn prompt mẫu để bắt đầu.",
                }
            ]
            st.session_state.last_run = None
            st.session_state.pending_request = None
            clear_chat_draft()
            st.rerun()

    return mode, engine_type, model_name, gemini_model, max_steps


def render_chat_panel():
    rendered_messages = []
    for message in st.session_state.messages:
        role = message["role"]
        label = "User" if role == "user" else "Bot / Agent"
        content = render_message_text(message["content"])
        summary = html.escape(message.get("summary", ""))
        details_html = ""
        if message.get("status") == "pending":
            details_html = """
            <div class="status-line">
                <span class="pulse"></span>
                Model đang thinking. Nếu là ReAct mode, agent sẽ gọi tool khi cần.
            </div>
            """
        elif message.get("run"):
            run = message["run"]
            trace_text = html.escape("\n\n".join(build_trace_lines(run.get("trajectory", []))) or "Không có trace cho lượt chạy này.")
            tools_text = html.escape("\n\n".join(build_tool_lines(run.get("tool_calls", []))))
            raw_json = html.escape(json.dumps(run, ensure_ascii=False, indent=2))
            details_html = f"""
            <div class="run-summary">{summary}</div>
            <details>
                <summary>Trace / tool calls</summary>
                <div class="detail-label">Tool calls</div>
                <pre>{tools_text}</pre>
                <div class="detail-label">Trace</div>
                <pre>{trace_text}</pre>
            </details>
            <details>
                <summary>Raw JSON</summary>
                <div class="detail-label">Tool calls</div>
                <pre>{raw_json}</pre>
            </details>
            """
        rendered_messages.append(
            f"""
            <div class="message-row {role}">
                <div class="message-bubble {role}">
                    <span class="message-label">{label}</span>
                    {content}
                    {details_html}
                </div>
            </div>
            """
        )

    chat_html = f"""
    <!doctype html>
    <html>
    <head>
        <style>
            :root {{
                --surface: #F7FBFF;
                --surface-2: #EDF5FA;
                --ink: #10202B;
                --muted: #5F7482;
                --accent: #0B3A5B;
                --shadow-dark: rgba(122, 151, 169, 0.34);
                --shadow-light: rgba(255, 255, 255, 0.95);
            }}
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: transparent;
                color: var(--ink);
            }}
            .chat-shell {{
                border: 1px solid rgba(255, 255, 255, 0.72);
                border-radius: 8px;
                background: var(--surface);
                padding: 0.7rem 0.85rem;
                height: 520px;
                overflow-y: auto;
                box-shadow: inset 7px 7px 18px rgba(122, 151, 169, 0.18), inset -7px -7px 18px rgba(255,255,255,0.9);
            }}
            .message-row {{
                display: flex;
                width: 100%;
                margin: 0.55rem 0;
            }}
            .message-row.user {{
                justify-content: flex-end;
            }}
            .message-row.assistant {{
                justify-content: flex-start;
            }}
            .message-bubble {{
                max-width: 72%;
                border-radius: 8px;
                padding: 0.65rem 0.8rem;
                color: var(--ink);
                line-height: 1.48;
                word-break: break-word;
                box-shadow: 6px 6px 14px rgba(122, 151, 169, 0.22), -6px -6px 14px rgba(255,255,255,0.95);
                font-size: 0.95rem;
            }}
            .message-bubble.user {{
                background: var(--accent);
                color: #FFFFFF;
                border-bottom-right-radius: 3px;
            }}
            .message-bubble.assistant {{
                background: var(--surface-2);
                border-bottom-left-radius: 3px;
            }}
            .message-bubble strong {{
                font-weight: 800;
            }}
            .message-bubble code {{
                border-radius: 5px;
                padding: 0.08rem 0.28rem;
                background: rgba(11, 58, 91, 0.10);
                color: var(--accent);
                font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
                font-size: 0.88em;
            }}
            .message-bubble.user code {{
                background: rgba(255, 255, 255, 0.18);
                color: #FFFFFF;
            }}
            .message-label {{
                display: block;
                font-size: 0.68rem;
                font-weight: 750;
                opacity: 0.78;
                margin-bottom: 0.22rem;
                text-transform: uppercase;
            }}
            .status-line, .run-summary {{
                margin-top: 0.55rem;
                padding-top: 0.45rem;
                border-top: 1px solid rgba(95,116,130,0.18);
                color: var(--muted);
                font-size: 0.82rem;
            }}
            .pulse {{
                display: inline-block;
                width: 0.55rem;
                height: 0.55rem;
                margin-right: 0.35rem;
                border-radius: 999px;
                background: var(--accent);
                animation: pulse 1.1s infinite ease-in-out;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 0.35; transform: scale(0.9); }}
                50% {{ opacity: 1; transform: scale(1.15); }}
            }}
            details {{
                margin-top: 0.45rem;
                font-size: 0.82rem;
            }}
            summary {{
                cursor: pointer;
                color: var(--accent);
                font-weight: 700;
            }}
            .detail-label {{
                margin-top: 0.45rem;
                color: var(--muted);
                font-size: 0.75rem;
                font-weight: 700;
                text-transform: uppercase;
            }}
            pre {{
                max-height: 180px;
                overflow: auto;
                white-space: pre-wrap;
                word-break: break-word;
                background: rgba(255,255,255,0.58);
                border-radius: 8px;
                padding: 0.55rem;
                color: var(--ink);
            }}
        </style>
    </head>
    <body>
        <div id="chat-shell" class="chat-shell">{"".join(rendered_messages)}</div>
        <script>
            const chatShell = document.getElementById("chat-shell");
            chatShell.scrollTop = chatShell.scrollHeight;
        </script>
    </body>
    </html>
    """
    components.html(
        chat_html,
        height=532,
        scrolling=False,
    )


def render_prompt_panel(mode, engine_type, openai_model, gemini_model, max_steps):
    pending_active = st.session_state.get("pending_request") is not None
    input_key = f"chat_draft_input_{st.session_state.chat_input_version}"
    if input_key not in st.session_state:
        st.session_state[input_key] = st.session_state.chat_draft
    with st.form("prompt_form", clear_on_submit=True):
        prompt = st.text_input(
            "Message",
            placeholder="Nhập tin nhắn cho chatbot...",
            label_visibility="collapsed",
            key=input_key,
            disabled=pending_active,
        )
        submitted = st.form_submit_button("Send", type="primary", use_container_width=True, disabled=pending_active)

    if submitted and prompt.strip():
        enqueue_prompt(prompt.strip(), mode, engine_type, openai_model, gemini_model, max_steps)
        st.rerun()

    st.caption("Prompt mẫu")
    sample_cols = st.columns(4)
    for idx, (label, prompt) in enumerate(SAMPLE_PROMPTS.items()):
        with sample_cols[idx]:
            st.button(label, use_container_width=True, on_click=set_chat_draft, args=(prompt,), disabled=pending_active)


def render_observability_panel():
    run = get_latest_run()
    st.markdown('<div class="panel-title">Run Observability</div>', unsafe_allow_html=True)

    if st.session_state.get("pending_request"):
        pending = st.session_state.pending_request
        st.info(f'{pending["mode"]} đang xử lý prompt trong chat session.')
        st.caption("Log chi tiết sẽ xuất hiện sau khi model hoàn tất.")
        render_tool_reference()
        return

    if not run:
        st.caption("Chưa có lượt chạy nào. Gửi prompt để xem thinking, observation và tool calls tại đây.")
        render_tool_reference()
        return

    run = {
        "status": run.get("status", "unknown"),
        "mode": run.get("mode", "unknown"),
        "thinking_count": run.get("thinking_count", 0),
        "tool_calls": run.get("tool_calls", []),
        "log_events": run.get("log_events", 0),
        "latency_ms": run.get("latency_ms", 0),
        "usage": ensure_total_tokens(run.get("usage", {})),
        "engine": run.get("engine", "unknown"),
        "model": run.get("model", "unknown"),
        "trajectory": run.get("trajectory", []),
    }

    m1, m2 = st.columns(2)
    m1.metric("Status", run["status"].upper())
    m2.metric("Mode", run["mode"])
    m3, m4 = st.columns(2)
    m3.metric("Thinking", f'{run["thinking_count"]} turns')
    m4.metric("Tool calls", len(run["tool_calls"]))
    m5, m6 = st.columns(2)
    m5.metric("Log events", run["log_events"])
    m6.metric("Latency", f'{run["latency_ms"]:.1f} ms')

    usage = run["usage"]
    st.markdown("#### Tokens")
    t1, t2, t3 = st.columns(3)
    t1.metric("Prompt", f'{usage["prompt_tokens"]:,}')
    t2.metric("Output", f'{usage["completion_tokens"]:,}')
    t3.metric("Total", f'{usage["total_tokens"]:,}')

    st.caption(f'Provider: {run["engine"]} | Model: {run["model"]}')

    if run["tool_calls"]:
        chips = "".join(
            f'<span class="tool-chip">{call.get("tool", "unknown")}</span>'
            for call in run["tool_calls"]
        )
        st.markdown(chips, unsafe_allow_html=True)

    st.markdown("#### Agent Workflow")
    if run["tool_calls"]:
        for call in run["tool_calls"][:3]:
            st.caption(f"Tool: {call.get('tool', 'unknown')}({compact_json(call.get('args', {}))})")
            if call.get("result") is not None:
                st.caption(f"Observation: {compact_json(call.get('result'))}")
    else:
        st.caption("No tool calls for this run.")

    st.markdown("#### Latest Trace")
    trace_lines = build_trace_lines(run.get("trajectory", []))
    if trace_lines:
        for line in trace_lines[:5]:
            st.caption(line)
        if len(trace_lines) > 5:
            st.caption(f"+ {len(trace_lines) - 5} dòng chi tiết trong chat bubble.")
    else:
        st.caption("Chatbot baseline không có ReAct trace.")

    render_tool_reference()


def render_tool_reference():
    with st.expander("Available tools", expanded=False):
        for spec in TOOL_SPECS:
            st.markdown(f"**`{spec['name']}`**: {spec['description']}")


init_state()
left, right = st.columns([0.66, 0.34], gap="medium")

with right:
    mode, engine_type, openai_model, gemini_model, max_steps = render_control_panel()
    render_observability_panel()

with left:
    st.markdown('<div class="eyebrow">ReAct Commerce Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="screen-title">Chat Session</div>', unsafe_allow_html=True)
    render_chat_panel()
    render_prompt_panel(mode, engine_type, openai_model, gemini_model, max_steps)

if st.session_state.get("pending_request"):
    completed = complete_pending_request()
    if completed:
        st.rerun()
