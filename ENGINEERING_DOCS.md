# ENGINEERING_DOCS.md — Technical Architecture & Developer Guide

## 📌 Overview
This document provides a comprehensive technical breakdown of the codebase, tech stack, system architecture, guardrails, and Web UI implementation for **Lab 03: Chatbot Baseline vs ReAct Agent**.

This repository implements an e-commerce assistant capable of executing a **Thought–Action–Observation** reasoning loop with ground-truth database tools, contrasting it against a single-call Chatbot baseline.

---

## 🛠️ 1. Tech Stack & Environment

| Component | Technology / Library | Purpose |
|---|---|---|
| **Language** | Python 3.12+ | Core application runtime |
| **Web UI** | Streamlit `1.30.0+` | Interactive Web Dashboard for Live Demo |
| **Testing** | Pytest | Unit testing for tools, baseline, agent V1, agent V2 |
| **LLM Integrations** | `google-generativeai`, `openai` | Live LLM APIs for Gemini and OpenAI |
| **Environment / Config** | `python-dotenv`, `.streamlit/config.toml` | Key management & headless server config |
| **Telemetry & Metrics** | Custom JSON Logger | System logging to `logs/execution.jsonl` |

### Environment Configuration
- **Virtual Environment**: Located at `.venv/` (Windows PowerShell executable path: `.\.venv\Scripts\python.exe`).
- **Module Resolution**: All submodules resolve from root directory `src/`. Scripts and `app.py` prepend `os.path.abspath(os.path.dirname(__file__))` to `sys.path`.
- **Streamlit Config**: `.streamlit/config.toml` enforces `headless = true` and `gatherUsageStats = false` for seamless background execution.

---

## 🏛️ 2. Core Architecture & Directory Structure

```
Day03-2A202601695-TrieuDuong/
├── app.py                      # Main Streamlit Web UI Application
├── run_demo.py                 # Streamlit CLI Launcher script
├── requirements.txt            # System dependencies
├── .streamlit/
│   └── config.toml             # Streamlit server settings
├── src/
│   ├── core/
│   │   ├── llm_provider.py     # Base abstract class for LLM Providers
│   │   ├── gemini_provider.py  # Google Gemini API integration
│   │   └── openai_provider.py  # OpenAI API integration
│   ├── chatbot/
│   │   └── chatbot.py          # Baseline Chatbot (1 LLM call, 0 tools)
│   ├── tools/
│   │   └── tools.py            # Ground Truth E-Commerce Tool Contracts & Mock Catalog
│   ├── agent/
│   │   ├── agent.py            # ReAct Agent V1 (CoT Loop & Action Parser)
│   │   └── agent_v2.py         # ReAct Agent V2 (Guardrails & Loop Prevention)
│   └── telemetry/
│       ├── logger.py           # Structured JSON Event Logger
│       └── metrics.py          # Latency & Token Efficiency metrics
├── scripts/
│   └── run_lab_evaluation.py   # Benchmark evaluation runner for 5 test cases
├── tests/
│   ├── test_chatbot_baseline.py
│   ├── test_tools.py
│   ├── test_agent_react_loop.py
│   └── test_agent_recovery.py
├── artifacts/
│   ├── evaluation/             # Benchmark JSON outputs
│   └── traces/                 # Agent reasoning trace JSON files
└── report/
    ├── group_report/           # Team report
    └── individual_reports/     # Student individual report
```

---

## 🔧 3. Tool Contracts & Mock Database (`src/tools/tools.py`)

Tools provide deterministic ground-truth data from the database. All tool functions return structured JSON dictionaries and handle missing parameters with explicit error signatures (`ok: false`).

### 1. `check_stock(item_name: str)`
- **Catalog**: `iPhone` (price: 25,000,000đ, stock: 15), `MacBook` (price: 35,000,000đ, stock: 0), `iPad` (price: 18,000,000đ, stock: 8).
- **Error**: Returns `{"ok": false, "error": "item_not_found"}` if item is not in catalog.

### 2. `get_discount(coupon_code: str)`
- **Coupons**: `WINNER` (valid: true, percent: 10%), `SUMMER` (valid: true, percent: 15%), `LEGACY` (valid: false, expired).
- **Error**: Returns `{"ok": false, "error": "invalid_coupon"}` if coupon is invalid or missing.

### 3. `calc_shipping(weight: float, destination: str)`
- **Rates**: `Hanoi` (base 30,000đ + 10,000đ/kg), `Saigon` (base 40,000đ + 15,000đ/kg), `Danang` (base 35,000đ + 12,000đ/kg).
- **Error**: Returns `{"ok": false, "error": "unsupported_destination"}` if location is unsupported.

---

## 🤖 4. Agent Architecture & ReAct State Machine

### Chatbot Baseline (`src/chatbot/chatbot.py`)
- **Protocol**: 1 LLM call, 0 tool access.
- **Limitation**: Cannot query real-time stock or shipping costs; susceptible to hallucinated pricing and stale facts.

### ReAct Agent V1 (`src/agent/agent.py`)
- **Loop**: `Thought -> Action -> Observation -> Thought -> ... -> Final Answer`.
- **Action Parsing**: Matches `Action: tool_name({"arg": "val"})`, `Action: tool_name(arg="val")`, or raw JSON.
- **Budget Control**: Enforces `max_steps` (default: 6). Falls back safely if step budget is exceeded.
- **Token Usage**: Aggregates `usage` from each `llm.generate()` call into `result["usage"]` (prompt/completion/total tokens).

### ReAct Agent V2 (`src/agent/agent_v2.py`)
Extends `ReActAgent` with 4 production-grade guardrails:
1. **Repeated-Action Detector**: Maintains `action_history` signatures (`tool_name:sorted_args`). If an identical action is called $\ge 2$ times continuously, blocks execution and injects corrective observation to break infinite loops.
2. **Evidence Gate**: Intercepts premature `Final Answer` responses on quantitative data queries (e.g. price/shipping calculations) if no tools have been invoked yet.
3. **Robust Codeblock Stripper**: Cleans markdown fences (```json ... ```) and quotes before regex parsing.
4. **Unknown Tool Recovery**: Catches calls to non-existent tools and returns available tool signatures to guide the LLM back on track.

---

## 💻 5. Web UI & State Management (`app.py`)

The Streamlit Web UI provides an interactive live demo allowing users to run and visualize reasoning traces.

### UI Layout & Controls
1. **Right-Side Control Panel**:
   - `Mode` switch: Select between `ReAct V2`, `ReAct V1`, or `Chatbot`.
   - `Mode` is rendered as a tab-style pill switch with distinct color accents for V2, V1, and Chatbot.
   - `Provider` switch: Select between `Offline`, `Gemini`, or `OpenAI`.
   - `Model` picker: Gemini options are `gemini-3.5-flash-lite` and `gemini-2.5-flash-lite`; OpenAI options are `gpt-4o-mini` and `text-embedding-3-small`. Offline mode uses `demo-scripted-llm`.
   - `Max steps` slider: Adjust ReAct reasoning budget from 1 to 10.
   - `Clear` button: Reset the current chatbot session and observability state.
   - Streamlit's default header/toolbar/deploy UI is hidden via CSS so the app reads as a single-purpose chatbot surface.
2. **Chatbot Session Window**:
   - The left side is the primary conversation area rendered with `st.chat_message`.
   - User prompts and assistant answers remain in `st.session_state.messages`, so the UI behaves like a normal chatbot session.
   - There is no separate "Kết quả Thực thi & Telemetry Dashboard" section; answers appear directly in the conversation flow.
   - The conversation shell uses a fixed viewport-relative height with internal scrolling so the desktop page fits in one screen.
   - The UI theme uses a white-blue Neumorphism style: soft raised panels, inset chat surfaces, and dark ocean-blue accents.
3. **Prompt Submission**:
   - Users can submit free-form prompts through a compact `st.form` prompt bar below the conversation window.
   - Preset prompt buttons only fill `st.session_state.chat_draft`; the LLM/ReAct loop runs only after the user clicks `Send`.
   - `Send` enqueues a pending request and appends a short in-chat "Đang suy nghĩ..." assistant bubble first. The next rerun completes the LLM/ReAct call and replaces that pending bubble with the final answer plus a compact trace summary.
4. **Right-Side Observability Panel**:
   - The latest run is summarized on the right with Status, Mode, Thinking turns, Tool Calls, Log Events, Latency, and Tokens.
   - The panel reads from `st.session_state.last_run` and falls back to the latest assistant message containing a `run` payload, so observability remains populated after reruns.
   - Trace, tool-call JSON, and raw run payload are organized into tabs (`Trace`, `Tools`, `Raw`).
   - Available tool specs are kept in a collapsible reference expander below the run panel.
5. **Token Usage Handling**:
   - **Token Usage** is normalized through `ensure_total_tokens()` so providers that only return prompt/completion tokens still display a correct total.

### Scripted Provider for Offline Mode (`DemoScriptedLLM`)
To enable 100% deterministic offline demonstrations without API keys, `app.py` embeds `DemoScriptedLLM`. It models step-by-step reasoning for all 5 benchmark cases:
- **Case 1 (QA)**: Direct answer without tool calls.
- **Case 3 (2 iPhone + WINNER + Hanoi)**: 4 steps (`check_stock` -> `get_discount` -> `calc_shipping` -> `Final Answer 45.038.000 VNĐ`).
- **Case 4 (MacBook + Saigon)**: 2 steps (`check_stock` -> `Out of stock Final Answer`).
- **Case 5 (iPad + LEGACY + Saigon)**: 4 steps (`check_stock` -> `get_discount` -> `calc_shipping` -> `Final Answer 18.047.500 VNĐ`).

### Trajectory Rendering
- **Conversation Window**: Rendered by `render_chat_panel()` from `st.session_state.messages`.
- **Observability Metrics**: Rendered by `render_observability_panel()` from the latest normalized run payload in `st.session_state.last_run`.
- **Thought / LLM Output**: Rendered in the `Trace` tab as per-step text blocks.
- **Action Call & Arguments**: Rendered in the `Trace` tab and `Tools` tab as structured JSON.
- **Observation (Ground Truth)**: Rendered next to the tool action inside each trace expander.
- **Guardrail Trigger**: Rendered as warning rows in the trace tab when V2 safety rules activate.
- **Final Answer**: Rendered as the assistant message in the main chat session.

---

## 🧪 6. Verification & Developer Workflow

### Running Unit Tests
Execute the full Pytest suite (17 passing tests):
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Running Benchmark Evaluation
Run evaluation across all 5 benchmark cases:
```powershell
.\.venv\Scripts\python.exe scripts/run_lab_evaluation.py
```
Output saved to: `artifacts/evaluation/raw_results.json`.

### Running Streamlit Web UI
Start the web dashboard locally:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Access in browser at: **`http://localhost:8501`**.

---

## 🔑 7. Key Notes for Next Coding Agent
1. **Chat Session State**: The UI stores chat messages in `st.session_state.messages`, the latest observability payload in `st.session_state.last_run`, and one optional pending request in `st.session_state.pending_request`.
2. **Input State**: The prompt box uses a versioned widget key (`chat_draft_input_{chat_input_version}`) so sample prompt fill and post-send clearing do not mutate a live Streamlit widget key.
3. **Model Selection**: Live providers now receive model names from the control-panel dropdown via `OpenAIProvider(..., model_name=openai_model)` or `GeminiProvider(..., model_name=gemini_model)`.
4. **Token Usage**: Both `ReActAgent.run()` and `ReActAgentV2.run()` return `result["usage"]` with aggregated `prompt_tokens`, `completion_tokens`, and `total_tokens`. UI metrics should read from the normalized payload.
5. **Tool Schema Modifications**: If adding new tools to `src/tools/tools.py`, remember to register them in `TOOL_SPECS` and `TOOL_REGISTRY`, and add corresponding unit test cases in `tests/test_tools.py`.
6. **Guardrail Adjustments**: ReAct V2 guardrails are implemented in `src/agent/agent_v2.py`. Override `run()` or `parse_action()` cleanly when extending guardrail capabilities.
