import os
import json
import pytest
from typing import List, Dict, Any, Optional, Generator
from src.core.llm_provider import LLMProvider
from src.agent.agent import ReActAgent
from src.agent.agent_v2 import ReActAgentV2

class ScriptedFailureLLMProvider(LLMProvider):
    def __init__(self, responses: List[str]):
        super().__init__(model_name="scripted-failure-llm")
        self.responses = responses
        self.call_index = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        if self.call_index >= len(self.responses):
            resp = "Final Answer: End of responses."
        else:
            resp = self.responses[self.call_index]
            self.call_index += 1

        return {
            "content": resp,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "latency_ms": 100.0
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        yield "Stream"

def test_v1_fails_on_unknown_tool_and_repeats():
    # Model attempts unknown tool search_product repeatedly
    responses = [
        'Thought: Search product catalog.\nAction: search_product({"q": "iPhone"})',
        'Thought: Search again.\nAction: search_product({"q": "iPhone"})',
        'Thought: Search again.\nAction: search_product({"q": "iPhone"})'
    ]
    provider1 = ScriptedFailureLLMProvider(responses)
    v1_agent = ReActAgent(llm=provider1, max_steps=3)
    res_v1 = v1_agent.run("Tìm iPhone")

    assert res_v1["status"] == "max_steps_exceeded"

    # Save failed trace artifact
    trace_dir = os.path.join("artifacts", "traces")
    os.makedirs(trace_dir, exist_ok=True)
    failed_trace_path = os.path.join(trace_dir, "failed_trace.json")
    with open(failed_trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "query": "Tìm iPhone",
            "failure_type": "unknown_tool_and_repeated_loop",
            "v1_status": res_v1["status"],
            "v1_steps": res_v1["steps"],
            "rca": {
                "user_input": "Tìm iPhone",
                "expected_path": "check_stock -> Final Answer",
                "actual_path": "search_product -> search_product -> search_product",
                "first_divergence": "Step 1: hallucinated search_product tool",
                "root_cause": "LLM hallucinated a non-existent tool search_product. V1 executor returned unknown_tool error, but LLM repeated identical call until step limit.",
                "fix": "Added repeated action detector and available_tools hint in Agent V2."
            },
            "trajectory": res_v1["trajectory"]
        }, f, indent=2, ensure_ascii=False)

    assert os.path.exists(failed_trace_path)

def test_v2_recovers_and_blocks_repeated_action():
    responses = [
        'Thought: Search product.\nAction: search_product({"q": "iPhone"})',
        'Thought: Search product again.\nAction: search_product({"q": "iPhone"})',
        'Thought: Now use valid check_stock tool.\nAction: check_stock({"item_name": "iPhone"})',
        'Thought: Conclude answer.\nFinal Answer: iPhone hiện có giá 25.000.000 VNĐ và còn 15 sản phẩm.'
    ]

    provider2 = ScriptedFailureLLMProvider(responses)
    v2_agent = ReActAgentV2(llm=provider2, max_steps=5, max_repeated_actions=2)
    res_v2 = v2_agent.run("Tìm iPhone")

    assert res_v2["status"] == "success"
    assert res_v2["recovered_from_error"] is True
    assert "25.000.000" in res_v2["final_answer"]
