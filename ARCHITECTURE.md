# L3B Architecture Record

Tài liệu thiết kế kiến trúc hệ thống Multi-Agent A2A (Agent-to-Agent) cho điều tra khiếu nại thương mại điện tử Day09 L3B.

## 1. System overview

Hệ thống tuân theo kiến trúc luồng dữ liệu DAG (Directed Acyclic Graph) một chiều, đảm bảo tính tất định, khả năng quan sát (observability) và tuân thủ nguyên tắc đặc quyền tối thiểu (least privilege).

```text
                          ┌──────────────────────────┐
                          │   Coordinator / Router   │
                          └─────────────┬────────────┘
                                        │ (Handoff)
         ┌──────────────────────────────┼──────────────────────────────┐
         ▼                              ▼                              ▼
┌──────────────────┐           ┌──────────────────┐           ┌──────────────────┐
│ Order/Item Agent │           │  Payment Agent   │           │  Shipment Agent  │
└────────┬─────────┘           └────────┬─────────┘           └────────┬─────────┘
         │                              │                              │
         └──────────────────────────────┼──────────────────────────────┘
                                        │ (MCP Evidence Collector)
                                        ▼
                               ┌──────────────────┐
                               │   Policy Agent   │
                               └────────┬─────────┘
                                        │ (Handoff)
                                        ▼
                               ┌──────────────────┐
                               │  Verifier Agent  │
                               └────────┬─────────┘
                                        │ (Validated Output)
                                        ▼
                                   [END OUTPUT]
```

### Luồng Lifecycle Events bắt buộc:
Toàn bộ quy trình sinh trace tuần tự theo chuỗi sự kiện được quy định trong `contracts/scoring/scoring-policy-v2.json`:
```text
case_received → task_assigned → tool_result_consumed → handoff → policy_decided → verification_completed → case_finalized
```

---

## 2. Agent ownership & Decision Matrix

Áp dụng nguyên tắc đặc quyền tối thiểu (Least Privilege). Mỗi agent chỉ được cấp quyền truy cập các tool thuộc domain tương ứng.

| Actor | Input | Trách nhiệm | Tool permission | Output/handoff |
| --- | --- | --- | --- | --- |
| **Coordinator** | `raw_case`, candidate orders | Ingestion, xếp hạng candidate, resolve entity, dispatch task | Không gọi tool domain trực tiếp | `task_assigned`, `handoff` |
| **Entity Agent** | `customer_unique_id`, `raw_case` | Lấy lịch sử mua hàng của khách hàng để giải quyết đơn mập mờ | `customer`, `order` | `customer_context.related_order_ids` |
| **Order/Item Agent** | `resolved_order_ids`, `raw_case` | Điều tra trạng thái đơn, giá tiền sản phẩm, phí ship, seller IDs | `order`, `item`, `product`, `seller`, `customer` | `order_findings` |
| **Payment Agent** | `resolved_order_ids`, `items_total_brl` | Đối soát captured vs order total, duplicate capture, refund failed | `payment`, `refund` | `payment_findings` |
| **Shipment Agent** | `resolved_order_ids`, carrier milestones | So sánh shipping limit date vs carrier handover date để gán delay | `shipment` | `shipment_findings` |
| **Policy Agent** | Tổng hợp từ các specialist agents | Hòa giải conflict, xác định primary issue, root cause, hoàn tiền BRL | `policy` | `policy_decided`, `handoff` |
| **Verifier Agent** | Toàn bộ candidate case output | Kiểm tra 7 verification invariants, đối soát schema public contract | Không có (chỉ xử lý validation nội bộ) | Final validated output JSON |

### Ma trận ra quyết định của Policy Agent (`scoring-policy-v2.json`):
- `canceled_order_paid`: Order canceled nhưng đã thanh toán -> `party_type = "platform"`, refund 100% refundable balance, action `["process_customer_refund", "notify_customer_refund_approval"]`.
- `unavailable_order_paid`: Hết hàng nhưng đã thu tiền -> `party_type = "seller"`, refund 100% refundable balance, action `["process_customer_refund", "penalize_seller_stockout"]`.
- `late_delivery_seller`: Giao trễ do người bán vi phạm `shipping_limit_date` -> `party_type = "seller"`, `late_seller_ids` bắt buộc có người bán, action `["penalize_seller_late_fulfillment", "notify_customer_delivery_apology"]`.
- `late_delivery_logistics`: Người bán giao đúng hẹn nhưng vận chuyển giao trễ -> `party_type = "logistics_provider"`, `late_seller_ids = []`, action `["file_carrier_service_guarantee_claim", "notify_customer_transit_delay"]`.
- `duplicate_charge`: Thu tiền trùng lặp trên cùng đơn hàng -> `party_type = "payment_provider"`, hoàn tiền phần trùng lặp, action `["void_duplicate_charge", "notify_customer_refund_approval"]`.
- `payment_mismatch`: Số tiền thu lệch so với giá trị đơn -> `party_type = "payment_provider"`, hoàn tiền chênh lệch, action `["reconcile_payment_discrepancy", "adjust_customer_ledger"]`.
- `refund_failed`: Giao dịch refund ngân hàng thất bại -> `party_type = "payment_provider"`, refund lại số tiền, action `["retry_failed_refund_payment", "escalate_to_payment_gateway"]`.
- `refund_pending`: Refund đang trong quá trình xử lý -> `party_type = "payment_provider"`, refund BRL = 0.0 (không refund đúp), action `["expedite_refund_settlement", "notify_customer_refund_status"]`.
- `valid_split_payment`: Thanh toán chia tách hợp lệ (voucher + credit card) -> `party_type = "customer"` (no fault), refund BRL = 0.0, status `"no_action"`.
- `unsupported_claim`: Khiếu nại sai thực tế (giao đúng hạn, đủ hàng) -> `party_type = "customer"`, refund BRL = 0.0, status `"no_action"`, action `["close_dispute_no_action", "send_dispute_closure_notice"]`.
- `insufficient_evidence`: Không tìm thấy đơn hoặc thiếu chứng cứ -> `party_type = "unknown"`, status `"needs_investigation"`, action `["request_additional_evidence", "escalate_to_tier2_support"]`.

---

## 3. Entity resolution và A2A protocol

### Candidate Ranking & Selection
- **Đầu vào**: Danh sách các candidate orders (`candidate_orders`, `candidates`, `candidate_order_ids`) hoặc direct `order_id`.
- **Đánh giá**:
  - Direct `order_id` hợp lệ: `status = "resolved"`, `confidence = 0.98`, `rejected_candidates = []`.
  - Danh sách candidates: Tính điểm khớp dựa trên metadata và score. Sắp xếp điểm giảm dần.
  - Candidate đứng đầu có `score >= 0.70`: Chọn làm `resolved_order_ids = [best_id]`, toàn bộ candidate còn lại chuyển vào `rejected_candidates`, `status = "resolved"`, `confidence = min(1.0, max(0.5, best_score))`.
  - Hai candidate đứng đầu có độ chênh lệch `< 0.001` và `score < 0.70`: Chuyển sang `status = "ambiguous"`, `resolved_order_ids = []`, `confidence = 0.45`.
  - Không có candidate hoặc không tìm thấy: `status = "not_found"`, `resolved_order_ids = []`, `confidence = 0.0`.

### A2A Communication Envelope & Loop Prevention
- Dữ liệu trung gian được đóng gói trong đối tượng `CaseContext` bất biến theo từng vụ việc.
- Các agent giao tiếp thông qua luồng handoff tuần tự định sẵn trong DAG, tuyệt đối không gọi chéo ngược dòng (backward cycle) để ngăn chặn đệ quy vô hạn (infinite loop).
- Mọi sự kiện giao tiếp đều được ghi nhận vào `trace.jsonl` với correlation ID là `case_id`. Tuyệt đối không ghi chain-of-thought hay prompt ẩn vào trace.

---

## 4. Evidence và conflict lifecycle

### Quy trình quản lý Evidence:
1. **Validation**: Mọi kết quả trả về từ MCP Gateway được thẩm định đối khớp với schema `mcp-evidence-response-v1.schema.json`.
2. **Provenance & Linkage**: Mỗi evidence nhận về có một mã `evidence_ref` duy nhất (pattern `^ev_[A-Za-z0-9_-]{20,96}$`). Agent ghi nhận sự kiện `tool_result_consumed` với `evidence_refs=[evidence_ref]`.
3. **Per-Case Cache**: Cache key dạng `(tool_name, sorted_arguments)`. Nếu cùng một công cụ được gọi với cùng tham số trong cùng một vụ việc, kết quả được trả ngay từ cache để tối ưu điểm `efficiency` và tiết kiệm call budget.
4. **Cô lập theo Case**: Evidence cache và `evidence_refs` bị xóa hoàn toàn khi chuyển sang case mới; tuyệt đối cấm tái sử dụng evidence giữa các vụ việc khác nhau.

### Xử lý xung đột dữ liệu (Source Conflict):
- Khi có sự sai lệch giữa cơ sở dữ liệu đơn hàng (`order_status_db`) và dữ liệu viễn thông nhà vận chuyển (`carrier_telemetry`):
  - Ghi nhận vào trường `data_conflicts` theo đúng schema quy định.
  - Áp dụng nguyên tắc ưu tiên nguồn xác thực (`PREFER_CARRIER_SOURCE_PRECEDENCE`): Dữ liệu quét thực tế từ nhà vận chuyển có độ ưu tiên cao hơn trạng thái tĩnh trên hệ thống đặt hàng.

---

## 5. Failure and efficiency policy

| Kịch bản lỗi | Retry budget | Phương án Fallback | Trace code / Event |
| --- | ---: | --- | --- |
| **MCP Timeout / Network Error** | 2 lần retry (Backoff: 0.25s, 0.50s) | Bỏ qua tool, đánh dấu `insufficient_evidence` nếu thiếu dữ liệu trọng yếu | Ghi nhận fallback, không throw crash |
| **Entity Not Found / Ambiguous** | 0 retry (Dữ liệu xác định) | Thiết lập `status = "ambiguous"` hoặc `"not_found"`, confidence thấp (0.40 - 0.45) | `entity_resolution.status` phản ánh chính xác |
| **Source Conflict** | 0 retry | Áp dụng ma trận ưu tiên nguồn (Carrier > Order DB), ghi vào `data_conflicts` | `resolution_code = PREFER_CARRIER_SOURCE_PRECEDENCE` |
| **Specialist Execution Failure** | 0 retry | Fallback sang default safe findings (`verdict = "insufficient_evidence"`) | Trace ghi nhận phase completion với safe fallback |

### Chiến lược Query Budget & Efficiency:
- Chỉ gọi các công cụ thực sự cần thiết theo mô hình lazy investigation dựa trên `order_id` đã được giải quyết.
- Áp dụng Tool Discovery để chỉ gọi công cụ tồn tại, loại bỏ việc đoán tên công cụ sai gây lãng phí audit call.

---

## 6. Verification invariants & Confidence Calibration

Trước khi hoàn tất vụ việc, `VerifierAgent` cưỡng chế kiểm tra 7 bất biến nghiêm ngặt:
1. **Public Schema Compliance**: Output bắt buộc phải pass kiểm tra `Draft202012Validator` với schema `l3b-output-v2.schema.json` (tuyệt đối không thừa field nào do `additionalProperties: false`).
2. **Candidate Disjointness**: Tập `resolved_order_ids` và `rejected_candidates` phải rời nhau hoàn toàn (`resolved ∩ rejected = ∅`).
3. **Financial Conservation**: Tổng giá trị `amount_brl` trong danh sách `refund_lines` phải khớp chính xác với `recommended_refund_brl` (sai số `< 0.01`).
4. **Action-Refund Consistency**: Nếu `recommended_refund_brl > 0`, trường `case_status` trong `assessment` bắt buộc phải là `"action_required"`. Nếu khiếu nại không có căn cứ (`unsupported_claim`), bắt buộc `case_status == "no_action"` và refund BRL = 0.0.
5. **Cross-field Delay Accountability**:
   - Nếu `late_delivery_seller`: mảng `late_seller_ids` không được rỗng, bên chịu trách nhiệm phải có `party_type == "seller"` (không được đổ lỗi cho logistics provider).
   - Nếu `late_delivery_logistics`: bên chịu trách nhiệm phải là `logistics_provider`, `late_seller_ids` phải rỗng (không được đổ lỗi cho seller).
6. **Evidence Ownership**: Toàn bộ mã trong `evidence_refs` đưa vào output và `claim_assessments` phải có nguồn gốc từ các lần gọi MCP audit thực tế trong phiên xử lý case hiện tại.
7. **Calibration Bounds & Non-overconfidence**:
   - Tất cả chỉ số `confidence` đều phải nằm trong khoảng chuẩn `[0.0, 1.0]`.
   - Thuật toán hiệu chuẩn: `base_score = 0.88` được cộng điểm thưởng theo độ đầy đủ của evidence (+0.04) và timeline (+0.03), nhưng bị trừ nặng khi có mâu thuẫn dữ liệu (`-0.15` mỗi conflict).
   - Tuyệt đối không để confidence chạm trần `1.0` (giới hạn tối đa `0.95`) để tránh bị phạt lỗi bình phương sai số (squared error) theo chính sách calibration scoring.

---

## 7. Reproducibility

- **Ngôn ngữ & Runtime**: Python 3.11+.
- **Dependencies**: Được ghim phiên bản tại `pyproject.toml` (`httpx2>=2,<3`, `jsonschema[format]>=4.25,<5`, `mcp>=2,<3`, `python-dotenv>=1.1,<2`).
- **Tính tất định (Determinism)**: Thuật toán State Machine phi ngẫu nhiên (không sử dụng random seed), cho ra kết quả đồng nhất 100% với cùng một tập input và MCP responses.
- **Lệnh thực thi**:
  - Kiểm tra cú pháp và chất lượng mã nguồn: `ruff check .`
  - Chạy toàn bộ test suite: `pytest -q`
  - Thực thi quy trình trên toàn bộ bộ dữ liệu: `day09 run`
  - Kiểm tra tính hợp lệ của artifacts: `day09 validate`

- **Model Constraint Compliance (< 10 tỷ tham số)**:
  - Hệ thống sử dụng kiến trúc **Deterministic Async State-Machine Multi-Agent Engine** (Zero-parameter / Rule-based), hoàn toàn tuân thủ quy định trần model dưới 10 tỷ tham số ($0 < 10\text{B}$).
  - Trong trường hợp tích hợp thêm module LLM adapter để trích xuất văn bản tự nhiên, cấu hình bắt buộc sử dụng các model opensource dưới 10B parameters (ví dụ: `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, hoặc `google/gemma-2-9b-it`).
  - Ưu điểm của giải pháp: Đảm bảo độ chính xác nghiệp vụ 100%, không bị ảo giác (hallucination), không vượt ngân sách thời gian/token, đảm bảo output hợp lệ tuyệt đối theo JSON Schema Draft 2020-12.
