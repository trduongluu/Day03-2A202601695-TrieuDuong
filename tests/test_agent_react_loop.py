import os
import json
import pytest
from typing import Dict, Any, List, Optional, Generator
from src.core.llm_provider import LLMProvider
from src.agent.agent import ReActAgent

class ScriptedLLMProvider(LLMProvider):
    def __init__(self, responses: List[str]):
        super().__init__(model_name="scripted-llm")
        self.responses = responses
        self.call_index = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        if self.call_index >= len(self.responses):
            resp = "Final Answer: No more scripted steps available."
        else:
            resp = self.responses[self.call_index]
            self.call_index += 1

        return {
            "content": resp,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            "latency_ms": 120.0
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        yield "Scripted stream"

def test_react_agent_multi_step_success():
    scripted_responses = [
        'Thought: Need to check stock and price for iPhone.\nAction: check_stock(item_name="iPhone")',
        'Thought: Need to verify discount coupon WINNER.\nAction: get_discount(coupon_code="WINNER")',
        'Thought: Need to calculate shipping fee to Hanoi.\nAction: calc_shipping(weight=0.8, destination="Hanoi")',
        'Thought: I have all required evidence.\nFinal Answer: Tổng chi phí cho 2 iPhone (giảm 10% coupon WINNER + phí ship Hà Nội 38.000đ) là 45.038.000 VNĐ.'
    ]

    provider = ScriptedLLMProvider(scripted_responses)
    agent = ReActAgent(llm=provider, max_steps=6)

    query = "Tôi muốn mua 2 iPhone, dùng mã 'WINNER' và giao tới Hà Nội. Khối lượng gói hàng 0.8 kg. Tổng tiền?"
    result = agent.run(query)

    assert result["status"] == "success"
    assert result["steps"] == 4
    assert len(result["tool_call_sequence"]) == 3
    assert result["tool_call_sequence"][0]["tool"] == "check_stock"
    assert result["tool_call_sequence"][1]["tool"] == "get_discount"
    assert result["tool_call_sequence"][2]["tool"] == "calc_shipping"
    assert "45.038.000" in result["final_answer"]

    # Save sanitized success trace artifact
    trace_dir = os.path.join("artifacts", "traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "success_trace.json")
    with open(trace_path, "w", encoding="utf-8") as f:
        json.dump({
            "query": query,
            "status": result["status"],
            "steps": result["steps"],
            "tool_call_sequence": result["tool_call_sequence"],
            "final_answer": result["final_answer"],
            "trajectory": result["trajectory"]
        }, f, indent=2, ensure_ascii=False)

    assert os.path.exists(trace_path)

def test_react_agent_max_steps_exceeded():
    infinite_loop_responses = [
        'Thought: Keep checking stock.\nAction: check_stock(item_name="iPhone")',
        'Thought: Keep checking stock.\nAction: check_stock(item_name="iPhone")',
        'Thought: Keep checking stock.\nAction: check_stock(item_name="iPhone")',
    ]
    provider = ScriptedLLMProvider(infinite_loop_responses)
    agent = ReActAgent(llm=provider, max_steps=2)

    result = agent.run("Check stock repeatedly")
    assert result["status"] == "max_steps_exceeded"
    assert result["steps"] == 2
    assert "Safe Fallback" in result["final_answer"]
