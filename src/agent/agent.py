import json
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.tools.tools import TOOL_REGISTRY, TOOL_SPECS

class ReActAgent:
    """
    ReAct Agent V1: Implements Thought-Action-Observation reasoning loop.
    """
    def __init__(self, llm: LLMProvider, tools: List[Dict[str, Any]] = None, max_steps: int = 5):
        self.llm = llm
        self.tools = tools if tools is not None else TOOL_SPECS
        self.max_steps = max_steps
        self.tool_registry = {t["name"]: t["function"] for t in self.tools if "function" in t}
        if not self.tool_registry:
            self.tool_registry = TOOL_REGISTRY

    def get_system_prompt(self) -> str:
        tool_descriptions = "\n".join([f"- {t['name']}: {t['description']}" for t in self.tools])
        return f"""You are a helpful e-commerce assistant. Solve customer queries step-by-step using tools.

AVAILABLE TOOLS:
{tool_descriptions}

FORMAT INSTRUCTIONS:
To use a tool, respond in the following format:
Thought: your line of reasoning.
Action: tool_name({{"arg_name": "value"}})

After receiving an Observation from a tool call, continue reasoning until you have sufficient facts to conclude:
Thought: I now have all required details.
Final Answer: your detailed response to the customer.

RULES:
1. Never invent or assume data (price, stock, shipping cost, discount). Always verify via tool calls.
2. Only output raw JSON args inside Action.
3. Do not output Observation yourself. Wait for the application to provide it.
"""

    def parse_action(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parses Action line:
        - Format A: Action: tool_name({"key": "value"})
        - Format B: Action: tool_name(key="value")
        - Format C: Action: {"tool": "tool_name", "args": {...}}
        """
        # Format C: Action: {"tool": "...", "args": {...}}
        json_match = re.search(r'Action:\s*(\{.*?\})', text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "tool" in data and "args" in data:
                    return data["tool"], data["args"]
            except json.JSONDecodeError:
                pass

        # Format A & B: Action: tool_name(...)
        pattern = r'Action:\s*([a-zA-Z0-9_]+)\s*\((.*?)\)'
        match = re.search(pattern, text, re.DOTALL)
        if not match:
            return None

        tool_name = match.group(1).strip()
        raw_args = match.group(2).strip()

        if not raw_args:
            return tool_name, {}

        # Try parsing raw_args as JSON
        try:
            parsed_json = json.loads(raw_args)
            if isinstance(parsed_json, dict):
                return tool_name, parsed_json
        except json.JSONDecodeError:
            pass

        # Try parsing kwarg format e.g. item_name="iPhone", destination="Hanoi"
        kwargs = {}
        kv_pairs = re.findall(r'([a-zA-Z0-9_]+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([0-9.]+))', raw_args)
        for key, val_str, val_str_single, val_num in kv_pairs:
            if val_str:
                kwargs[key] = val_str
            elif val_str_single:
                kwargs[key] = val_str_single
            elif val_num:
                kwargs[key] = float(val_num) if '.' in val_num else int(val_num)

        if kwargs:
            return tool_name, kwargs

        return tool_name, {}

    def parse_final_answer(self, text: str) -> Optional[str]:
        match = re.search(r'Final Answer:\s*(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tool_registry:
            return {
                "ok": False,
                "error": "unknown_tool",
                "message": f"Tool '{tool_name}' is not in available tools list.",
                "available_tools": list(self.tool_registry.keys())
            }

        try:
            fn = self.tool_registry[tool_name]
            return fn(**args)
        except Exception as e:
            return {
                "ok": False,
                "error": "execution_error",
                "message": f"Exception executing {tool_name}: {str(e)}"
            }

    def run(self, user_input: str) -> Dict[str, Any]:
        start_time = time.time()
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        system_prompt = self.get_system_prompt()
        trajectory = []
        messages = [
            f"System: {system_prompt}",
            f"User: {user_input}"
        ]

        steps = 0
        final_answer = None
        tool_call_sequence = []
        total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while steps < self.max_steps:
            steps += 1
            current_prompt = "\n\n".join(messages)
            llm_res = self.llm.generate(current_prompt)
            llm_text = llm_res.get("content", "")

            usage = llm_res.get("usage") or {}
            total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
            total_usage["total_tokens"] += usage.get("total_tokens", 0)

            trajectory.append({
                "step": steps,
                "type": "llm_output",
                "content": llm_text
            })

            # Check for Final Answer first
            fa = self.parse_final_answer(llm_text)
            if fa:
                final_answer = fa
                logger.log_event("AGENT_FINAL_ANSWER", {"step": steps, "final_answer": fa})
                break

            # Check for Action
            action_parsed = self.parse_action(llm_text)
            if action_parsed:
                tool_name, args = action_parsed
                tool_call_sequence.append({"tool": tool_name, "args": args})

                obs_result = self._execute_tool(tool_name, args)
                obs_str = f"Observation: {json.dumps(obs_result, ensure_ascii=False)}"

                trajectory.append({
                    "step": steps,
                    "type": "action_and_observation",
                    "tool": tool_name,
                    "args": args,
                    "observation": obs_result
                })

                messages.append(llm_text)
                messages.append(obs_str)
                logger.log_event("AGENT_STEP", {"step": steps, "tool": tool_name, "args": args, "obs": obs_result})
            else:
                # LLM outputted thought without action or final answer
                messages.append(llm_text)
                messages.append("Observation: Please proceed by outputting Action: tool_name(args) or Final Answer: <response>.")

        latency_ms = (time.time() - start_time) * 1000

        if not final_answer:
            final_answer = "Safe Fallback: Maximum reasoning steps reached before producing a Final Answer."
            status = "max_steps_exceeded"
        else:
            status = "success"

        # Derive total_tokens if provider only returned prompt/completion counts
        if total_usage["total_tokens"] == 0:
            total_usage["total_tokens"] = total_usage["prompt_tokens"] + total_usage["completion_tokens"]

        result = {
            "final_answer": final_answer,
            "status": status,
            "steps": steps,
            "tool_call_sequence": tool_call_sequence,
            "trajectory": trajectory,
            "latency_ms": latency_ms,
            "usage": total_usage
        }

        logger.log_event("AGENT_END", {"status": status, "steps": steps, "latency_ms": latency_ms, "usage": total_usage})
        return result
