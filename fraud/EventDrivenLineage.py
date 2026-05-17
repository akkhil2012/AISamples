# ── 7. Event-driven data lineage capture ─────────────────────
# IBM: event-driven lineage architecture — 70% traceability improvement
import json, datetime
 
lineage_log = []
 
def emit_lineage_event(source: str, target: str, operation: str, metadata: dict):
    event = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "source":    source,
        "target":    target,
        "operation": operation,
        "metadata":  metadata,
    }
    lineage_log.append(event)
    print(json.dumps(event, indent=2))
 
emit_lineage_event(
    source="raw_transactions",
    target="fraud_features",
    operation="TRANSFORM",
    metadata={"rows": 40_000_000, "pipeline": "spark_etl_v3"},
)
