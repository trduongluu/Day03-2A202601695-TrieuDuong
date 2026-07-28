import pytest
from typing import Dict, Any, Generator, Optional
from src.core.llm_provider import LLMProvider
from src.chatbot.chatbot import ChatbotBaseline

class MockLLMProvider(LLMProvider):
    def __init__(self, mock_response: str):
        super().__init__(model_name="mock-llm")
        self.mock_response = mock_response
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        self.call_count += 1
        return {
            "content": self.mock_response,
            "usage": {"prompt_tokens": 15, "completion_tokens": 20},
            "latency_ms": 50.0
        }

    def stream(self, prompt: str, system_prompt: Optional[str] = None) -> Generator[str, None, None]:
        yield self.mock_response

def test_chatbot_static_qa():
    provider = MockLLMProvider("Chính sách đổi trả của chúng tôi là 30 ngày cho các sản phẩm còn nguyên tem.")
    bot = ChatbotBaseline(provider)
    result = bot.chat("Chính sách đổi trả là gì?")
    
    assert result["tool_calls_count"] == 0
    assert provider.call_count == 1
    assert "30 ngày" in result["response"]

def test_chatbot_multistep_query_lack_of_evidence():
    # Chatbot baseline guesses an answer without tool evidence
    provider = MockLLMProvider("Tổng tiền 2 iPhone giao Hà Nội dùng mã WINNER khoảng 45.000.000 VNĐ.")
    bot = ChatbotBaseline(provider)
    result = bot.chat("Tôi muốn mua 2 iPhone, dùng mã WINNER và giao tới Hà Nội. Tổng tiền là bao nhiêu?")
    
    # Check baseline properties: strictly 1 LLM call, 0 tool calls
    assert result["tool_calls_count"] == 0
    assert provider.call_count == 1
    assert "45.000.000" in result["response"]
