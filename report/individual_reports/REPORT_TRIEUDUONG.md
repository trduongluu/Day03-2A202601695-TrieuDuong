# Individual Report: Lab 3 - Chatbot vs ReAct Agent

- **Student Name**: Triệu Dương
- **Student ID**: 2A202601695
- **Date**: 2026-07-28

---

## I. Technical Contribution (15 Points)

Trong bài lab này, tôi đã trực tiếp thiết kế, triển khai, tối ưu hóa và đánh giá toàn bộ các module trong hệ thống từ Chatbot Baseline, ReAct Agent V1/V2, hệ thống Telemetry, cho đến giao diện Live Demo Web UI chuyên nghiệp và tài liệu kỹ thuật dự án:

- **Modules Implemented**:
  1. `src/chatbot/chatbot.py`: Triển khai Chatbot Baseline chuẩn 1 LLM Call, 0 Tool Calls.
  2. `src/tools/tools.py`: Xây dựng 3 deterministic tool functions (`check_stock`, `get_discount`, `calc_shipping`), Tool Registry và Catalog dữ liệu mẫu với phản hồi JSON chuẩn hóa.
  3. `src/agent/agent.py`: Lắp ráp ReAct Agent V1 triển khai vòng lặp suy luận State Machine (`Thought -> Action -> Observation`), tích hợp bộ gom đếm Token Usage (`prompt_tokens`, `completion_tokens`, `total_tokens`) tự động tích lũy qua từng bước.
  4. `src/agent/agent_v2.py`: Nâng cấp ReAct Agent V2 bổ sung 4 Guardrails sản xuất (*Repeated-Action Detector*, *Evidence Gate*, *Markdown Codeblock Stripper*, *Unknown Tool Recovery*) và tích hợp Token Usage tracking.
  5. `app.py`: Thiết kế và phát triển giao diện Streamlit Live Demo UI theo phong cách Neumorphism hiện đại. Tách biệt hai vùng chức năng: Chat Session tương tác tự nhiên ở bên trái và Control & Observability Panel hiển thị suy luận, telemetry, trace, tool calls, token usage ở bên phải. Hỗ trợ đa mô hình (Gemini `gemini-3.5-flash-lite`/`gemini-2.5-flash-lite`, OpenAI `gpt-4o-mini`, Offline `demo-scripted-llm`).
  6. `ENGINEERING_DOCS.md`: Soạn thảo tài liệu kỹ thuật chi tiết về kiến trúc hệ thống, hợp đồng dữ liệu của Tool, quy tắc quản lý State trong Streamlit và hướng dẫn cho lập trình viên/coding agent tiếp nối công việc.
  7. `scripts/run_lab_evaluation.py`: Viết script benchmark tự động cho 5 test cases và xuất file kết quả định lượng.
  8. Pytest Test Suite (`tests/test_chatbot_baseline.py`, `tests/test_tools.py`, `tests/test_agent_react_loop.py`, `tests/test_agent_recovery.py`).

- **Code Highlights**:
  - **Tích lũy & Đếm Token tự động trong Agent (`src/agent/agent.py` & `src/agent/agent_v2.py`)**:
    ```python
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    while steps < self.max_steps:
        llm_res = self.llm.generate(current_prompt)
        usage = llm_res.get("usage") or {}
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)
    ```
  - **Cơ chế Phát hiện & Chặn lệnh lặp vô hạn (Repeated-Action Detector)**:
    ```python
    action_signature = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
    action_history.append(action_signature)
    if action_history.count(action_signature) >= self.max_repeated_actions:
        obs_result = {"ok": False, "error": "repeated_action_blocked", "message": "Action repeated..."}
        # Injected as observation to break infinite loop
    ```
  - **Quản lý Session State & Chuẩn hóa Token trong Web UI (`app.py`)**:
    ```python
    def ensure_total_tokens(usage: Optional[Dict[str, int]]) -> Dict[str, int]:
        usage = usage or {}
        p = usage.get("prompt_tokens", 0)
        c = usage.get("completion_tokens", 0)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": usage.get("total_tokens", p + c)}
    ```

- **Documentation**:
  - Mã nguồn tương tác chặt chẽ với luồng ReAct: Mỗi bước LLM đưa ra `Action`, Agent thực thi hàm tương ứng trong `TOOL_REGISTRY`, lấy kết quả `Observation` đưa ngược lại làm ngữ cảnh cho bước suy luận `Thought` tiếp theo cho đến khi đạt được `Final Answer`.
  - Toàn bộ thiết kế kỹ thuật, hợp đồng tool và quy tắc đồng bộ state UI được tài liệu hóa chi tiết tại [ENGINEERING_DOCS.md](file:///c:/Users/ADMIN/Desktop/Code%20Space/Day03-2A202601695-TrieuDuong/ENGINEERING_DOCS.md).

---

## II. Debugging Case Study (10 Points)

- **Problem Description**: 
  1. Trong phiên bản Agent V1 ban đầu, khi mô hình trả về lệnh gọi tool không tồn tại (hallucinated tool như `search_product`), Agent V1 trả về thông báo lỗi `unknown_tool`, nhưng ở bước tiếp theo LLM tiếp tục tạo lặp lại chính xác lệnh `Action: search_product({"q": "iPhone"})`. Việc này làm rơi vào vòng lặp vô hạn và tiêu tốn toàn bộ budget `max_steps` mà không có câu trả lời cuối cùng.
  2. Đồng thời, khi hiển thị Telemetry trên UI, một số Provider chỉ trả về `prompt_tokens` và `completion_tokens` mà không có giá trị `total_tokens`, dẫn đến thông số token hiển thị bằng 0 trên bảng điều khiển.

- **Log Source**: 
  - Trace thất bại được ghi nhận tại `artifacts/traces/failed_trace.json`:
    ```json
    {
      "failure_type": "unknown_tool_and_repeated_loop",
      "first_divergence": "Step 1: hallucinated search_product tool",
      "v1_status": "max_steps_exceeded"
    }
    ```
  - Log Telemetry trong `logs/execution.jsonl` phản ánh số token bị ngắt quãng.

- **Diagnosis**: 
  1. System Prompt của Agent V1 liệt kê danh sách tool nhưng chưa khẳng định quy tắc đóng "Chỉ được sử dụng các tool có trong danh sách".
  2. Executor V1 khi báo lỗi chưa trả về thông báo phản hồi bổ sung danh sách tool hợp lệ (`available_tools`) để hướng dẫn mô hình quay lại đúng hướng.
  3. Agent V1 thiếu bộ nhớ theo dõi các hành động đã thực hiện (`action_history`) để phát hiện lặp lại.
  4. Cấu trúc thống kê token giữa các LLM Provider chưa được tính tổng dồn qua nhiều lượt gọi LLM trong cùng một Agent execution run.

- **Solution**:
  - Triển khai `ReActAgentV2` trong `src/agent/agent_v2.py` tích hợp `action_history` với ngưỡng `max_repeated_actions=2`.
  - Bổ sung cơ chế inject Observation phản hồi khi gặp lỗi tool hoặc lặp hành động, yêu cầu LLM thay đổi tool hoặc chốt `Final Answer`.
  - Cập nhật cả `ReActAgent` và `ReActAgentV2` tự động tính tổng dồn `usage` (bao gồm `prompt_tokens`, `completion_tokens`, `total_tokens`) sau mỗi vòng lặp `llm.generate()`.
  - Bổ sung hàm chuẩn hóa `ensure_total_tokens()` trên UI `app.py` để đảm bảo telemetry hiển thị số lượng token chính xác.
  - Viết Unit Test `tests/test_agent_recovery.py` xác nhận Agent V2 tự hồi phục thành công từ lỗi lặp tool.

---

## III. Personal Insights: Chatbot vs ReAct (10 Points)

1. **Reasoning**: Khối `Thought` đóng vai trò là "vùng nháp suy luận" (scratchpad) giúp Agent phân tích bài toán, xác định các thông tin còn thiếu và lập kế hoạch đa bước trước khi quyết định gọi tool hay trả lời. So với Chatbot trả lời trực tiếp trong 1 lượt (dễ bị ảo giác giá cả/tồn kho), `Thought` giúp Agent phân tách yêu cầu thành các bước kiểm tra có bằng chứng thực tế từ database.
2. **Reliability**: Trong các câu hỏi Q&A tĩnh đơn giản (như hỏi về chính sách đổi trả hay giờ làm việc), Agent thực tế chạy **chậm hơn và tốn nhiều token hơn** so với Chatbot Baseline do chi phí suy luận dồn qua các bước (`Thought`). Chi phí suy luận của Agent chỉ thực sự hiệu quả và cần thiết đối với các tác vụ mua sắm phức tạp đòi hỏi tính toán logic và xác minh dữ liệu thực tế (*Ground Truth*).
3. **Observation**: Phản hồi từ môi trường (`Observation`) giúp Agent điều chỉnh hành vi tức thì. Ví dụ: Khi gọi `check_stock({"item_name": "MacBook"})` và nhận Observation `stock: 0`, Agent nhận biết ngay sản phẩm hết hàng và kết luận ngay `Final Answer` dừng cuộc hội thoại thay vì tiếp tục tính phí ship hay áp mã giảm giá một cách vô ích.

---

## IV. Future Improvements (5 Points)

- **Scalability**: Triển khai kiến trúc **Async / Parallel Tool Execution** để các tool độc lập như `check_stock` và `get_discount` có thể chạy song song trong cùng một lượt suy luận. Điều này giúp giảm latency tổng thể khi workflow mở rộng sang nhiều bước kiểm tra sản phẩm, coupon, vận chuyển và thanh toán.
- **Safety & Validation**: Bổ sung lớp **Tool Argument Validator / Guardrail Layer** trước khi thực thi tool để kiểm tra tham số đầu vào như số lượng âm, cân nặng không hợp lệ, coupon code bất thường hoặc destination không hỗ trợ. Với các hệ thống thật, lớp này cần chặn prompt injection và chỉ cho phép gọi các tool đã đăng ký trong `TOOL_REGISTRY`.
- **Conversation Memory**: Mở rộng Agent để hỗ trợ memory theo phiên hội thoại. Ví dụ, nếu người dùng đã hỏi “tôi muốn mua 2 iPhone” rồi tiếp tục hỏi “ship tới Hà Nội thì sao?”, Agent cần hiểu ngữ cảnh trước đó thay vì yêu cầu nhập lại toàn bộ thông tin.
- **Observability & Debugging UX**: Chuẩn hóa trace theo schema ổn định gồm `thought`, `action`, `args`, `observation`, `guardrail`, `latency_ms`, `usage`. UI nên hiển thị bản tóm tắt ngắn trong chat bubble và cho phép mở rộng để xem full trace khi cần debug.
- **Evaluation Coverage**: Bổ sung benchmark cho các case lỗi như coupon không tồn tại, destination không hỗ trợ, tool trả về `ok=false`, model gọi sai tool, model trả lời final answer quá sớm. Các case này giúp đo độ ổn định của ReAct Agent V2 so với Chatbot Baseline.
- **Performance / Tool Retrieval**: Khi số lượng tool tăng lên hàng trăm, áp dụng **Semantic Tool Retrieval (RAG for Tools)** để chỉ đưa các tool spec liên quan vào system prompt. Cách này giảm token cost, giảm nhiễu trong prompt và hạn chế khả năng model chọn sai tool.
