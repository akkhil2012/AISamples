# ── 8. Lineage-driven API with FastAPI ───────────────────────
# JPMC: lineage-driven API for end-to-end traceability
from fastapi import FastAPI, Request
import uuid, datetime
 
app = FastAPI()
audit_trail = []
 
@app.middleware("http")
async def lineage_middleware(request: Request, call_next):
    trace_id = str(uuid.uuid4())
    start = datetime.datetime.utcnow()
    response = await call_next(request)
    audit_trail.append({
        "trace_id":   trace_id,
        "path":       request.url.path,
        "method":     request.method,
        "status":     response.status_code,
        "duration_ms": (datetime.datetime.utcnow() - start).microseconds // 1000,
    })
    response.headers["X-Trace-Id"] = trace_id
    return response
 
@app.get("/score")
def score(): return {"fraud_prob": 0.12}