import time
from typing import Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

class ChatbotBaseline:
    """
    Chatbot Baseline implementation:
    - Exactly 1 LLM Call.
    - Zero tool calls (tool_calls = 0).
    - No tools embedded in prompt.
    - Direct response based on model internal knowledge / prompt context.
    """
    def __init__(self, provider: LLMProvider, system_prompt: Optional[str] = None):
        self.provider = provider
        self.system_prompt = system_prompt or (
            "You are a helpful e-commerce customer support assistant for an electronics store in Vietnam. "
            "Answer customer questions politely and accurately based on general store knowledge. "
            "If you do not know exact real-time inventory, live discounts, or real-time shipping fees, "
            "provide a polite general answer or state clearly what information is needed."
        )

    def chat(self, user_input: str) -> Dict[str, Any]:
        """
        Executes a single LLM call baseline protocol.
        Returns:
            Dict containing:
            - response: str
            - tool_calls_count: int (always 0)
            - usage: Dict[str, int]
            - latency_ms: float
        """
        start_time = time.time()
        logger.log_event("CHATBOT_START", {"input": user_input, "model": self.provider.model_name})

        llm_response = self.provider.generate(user_input, system_prompt=self.system_prompt)

        latency_ms = (time.time() - start_time) * 1000
        content = llm_response.get("content", "")
        usage = llm_response.get("usage", {"prompt_tokens": 0, "completion_tokens": 0})

        logger.log_event("CHATBOT_END", {
            "response": content,
            "tool_calls_count": 0,
            "latency_ms": latency_ms,
            "usage": usage
        })

        return {
            "response": content,
            "tool_calls_count": 0,
            "usage": usage,
            "latency_ms": latency_ms
        }
