import json
import re
import time
from typing import List, Dict, Any, Optional
from src.agent.agent import ReActAgent
from src.telemetry.logger import logger

class ReActAgentV2(ReActAgent):
    """
    ReAct Agent V2: Improved Agent with Guardrails:
    1. Repeated-Action Detection & Prevention (prevents infinite loops on identical tool calls).
    2. Robust Action JSON Parsing (handles markdown fences, single quotes).
    3. Unknown Tool Recovery & Corrective Guidance.
    4. Premature Final Answer Detection (Evidence Gate).
    """
    def __init__(self, llm, tools=None, max_steps: int = 5, max_repeated_actions: int = 2):
        super().__init__(llm, tools, max_steps)
        self.max_repeated_actions = max_repeated_actions

    def parse_action(self, text: str):
        """
        Robust Action Parser for V2:
        Extracts tool name and JSON arguments even if wrapped in ```json ... ``` or single quotes.
        """
        cleaned_text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)

        action = super().parse_action(cleaned_text)
        if action:
            return action

        pattern = r'Action:\s*([a-zA-Z0-9_]+)\s*:\s*(\{.*?\})'
        match = re.search(pattern, cleaned_text, re.DOTALL)
        if match:
            tool_name = match.group(1).strip()
            raw_json = match.group(2).strip().replace("'", '"')
            try:
                args = json.loads(raw_json)
                return tool_name, args
            except json.JSONDecodeError:
                pass

        return None

    def run(self, user_input: str) -> Dict[str, Any]:
        start_time = time.time()
        logger.log_event("AGENT_V2_START", {"input": user_input, "model": self.llm.model_name})

        system_prompt = self.get_system_prompt()
        trajectory = []
        messages = [
            f"System: {system_prompt}",
            f"User: {user_input}"
        ]

        steps = 0
        final_answer = None
        tool_call_sequence = []
        action_history = []
        recovered_from_error = False
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

            # Check Final Answer
            fa = self.parse_final_answer(llm_text)
            if fa:
                query_lower = user_input.lower()
                is_data_query = any(
                    k in query_lower
                    for k in ["giá", "tổng", "tồn kho", "ship", "phí", "total", "price", "shipping", "discount"]
                )
                if is_data_query and len(tool_call_sequence) == 0:
                    messages.append(llm_text)
                    messages.append("Observation: Error: You provided a Final Answer without retrieving ground-truth evidence from available tools. Please call appropriate tool first.")
                    trajectory.append({
                        "step": steps,
                        "type": "guardrail_trigger",
                        "guardrail": "evidence_gate",
                        "message": "Blocked premature final answer"
                    })
                    continue

                final_answer = fa
                logger.log_event("AGENT_V2_FINAL_ANSWER", {"step": steps, "final_answer": fa})
                break

            # Check Action
            action_parsed = self.parse_action(llm_text)
            if action_parsed:
                tool_name, args = action_parsed

                action_signature = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
                action_history.append(action_signature)
                repeat_count = action_history.count(action_signature)

                if repeat_count >= self.max_repeated_actions:
                    obs_result = {
                        "ok": False,
                        "error": "repeated_action_blocked",
                        "message": f"Action '{tool_name}' with identical arguments was repeated {repeat_count} times. Please choose a different tool or formulate Final Answer."
                    }
                    obs_str = f"Observation: {json.dumps(obs_result, ensure_ascii=False)}"
                    trajectory.append({
                        "step": steps,
                        "type": "guardrail_trigger",
                        "guardrail": "repeated_action_detector",
                        "tool": tool_name,
                        "observation": obs_result
                    })
                    messages.append(llm_text)
                    messages.append(obs_str)
                    recovered_from_error = True
                    continue

                # Execute Tool
                obs_result = self._execute_tool(tool_name, args)
                if not obs_result.get("ok", True):
                    recovered_from_error = True

                tool_call_sequence.append({"tool": tool_name, "args": args, "result": obs_result})
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
            else:
                messages.append(llm_text)
                messages.append("Observation: Error: Could not parse Action or Final Answer. Format strictly: Action: tool_name({\"arg\": \"val\"}) or Final Answer: <response>.")
                recovered_from_error = True

        latency_ms = (time.time() - start_time) * 1000

        if not final_answer:
            final_answer = "Safe Fallback (V2): Reached max steps or blocked by safety guardrails."
            status = "safe_fallback"
        else:
            status = "success"

        # Derive total_tokens if provider only returned prompt/completion counts
        if total_usage["total_tokens"] == 0:
            total_usage["total_tokens"] = total_usage["prompt_tokens"] + total_usage["completion_tokens"]

        return {
            "final_answer": final_answer,
            "status": status,
            "steps": steps,
            "tool_call_sequence": tool_call_sequence,
            "trajectory": trajectory,
            "recovered_from_error": recovered_from_error,
            "latency_ms": latency_ms,
            "usage": total_usage
        }
