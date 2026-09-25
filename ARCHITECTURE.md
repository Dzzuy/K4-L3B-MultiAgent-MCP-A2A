# L3B Architecture Record

## System overview

```text
Input Case → Entity/Customer → Order/Product → Coordinator
                    customer history          ├→ Shipment
                    candidate verification    └→ Payment/Refund
                                                  ↓
                               Policy rules + optional Qwen audit → Verifier → L3B JSON
```

Shipment and payment agents run concurrently after order resolution.

## Agent ownership

| Agent | MCP tools |
| --- | --- |
| entity/customer | `get_customer_history`, candidate `get_order` |
| order/product | `get_order`, `get_order_items`, `get_product_context`, conditional `get_sellers` |
| shipment | `get_shipment_summary` |
| payment/refund | `get_order_payments`, `get_payment_timeline`, conditional `get_refund_timeline` |
| policy | `get_policy` |
| verifier | none |

## Entity and evidence lifecycle

Input hints and claimed orders are not ground truth. Customer history independently links orders; rejected-candidate evidence cannot support a selected order. Multiple valid candidates remain ambiguous: there is no latest-order heuristic.

The gateway validates an envelope, then `Evidence` preserves its verbatim `evidence_ref` and internal call arguments. `tool_result_consumed` is emitted, identical calls use a per-case cache, policy selects required tools for the selected entity, and verifier filters again. Evidence is never reused across cases.

## Facts, policy, and verification

Rows outside the lifecycle window are excluded. Payment rows reconcile copied captures, and split payments are not duplicates. Seller deadlines are seller-specific; overdue undelivered orders can be late; authoritative order dates override misleading shipment events.

Missing policy means insufficient evidence. Supported secondary claims remain supported. Pending/failed refunds are capped by outstanding capture. Seller responsibility is scoped to affected and late sellers. Verifier checks provenance, selected-order scope, policy evidence, finance totals, seller scope, confidence, and the final L3B schema.

Qwen/OpenRouter is optional and advisory only. Deterministic evidence-backed rules remain authoritative: audit disagreement cannot alter the issue, refund, evidence, or responsibility.

## Failure, efficiency, and reproducibility

Tool retries and gateway reconnects are bounded. Refund/seller calls are conditional, product context follows scope, and the system avoids brute-force scans and infinite retries.

```bash
python -m pip install -e ".[dev]"
day09 validate-inputs
day09 mcp-tools
day09 run
day09 validate
day09 package --output dist/submission.zip
```
