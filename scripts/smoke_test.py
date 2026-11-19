"""End-to-end smoke test against a RUNNING server (real HTTP).

    python scripts/smoke_test.py --base-url http://127.0.0.1:8000 [--no-llm]

Steps: readiness -> ingest a sample document -> ask a question whose answer is in it ->
check the answer is grounded and the citations point at the right file/section/text ->
(restart check is manual: see README, "Verify persistence").
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

SAMPLE = b"""# Payment Service Runbook

## Retry policy

Failed payment requests are retried up to 4 times using exponential backoff
starting at 250 ms, with random jitter. After the final attempt the request
is moved to the dead-letter queue named payments-dlq.

## Deployment

The service is deployed with a Kubernetes rolling update; maxUnavailable is 1.
"""


def check(condition: bool, label: str) -> None:
    print(("PASS  " if condition else "FAIL  ") + label)
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--no-llm", action="store_true", help="only verify retrieval (generate=false)")
    parser.add_argument("--wait", type=int, default=120, help="seconds to wait for /ready")
    args = parser.parse_args()
    http = httpx.Client(base_url=args.base_url, timeout=180)

    deadline = time.time() + args.wait
    while True:
        try:
            ready = http.get("/ready")
            if ready.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        check(time.time() < deadline, "server became ready") if time.time() >= deadline else time.sleep(2)
    check(True, "server is ready")

    up = http.post("/api/v1/documents", files={"file": ("payment-runbook.md", SAMPLE, "text/markdown")})
    check(
        up.status_code == 200 and up.json()["status"] in {"indexed", "unchanged"},
        f"document ingested ({up.json().get('status')})",
    )

    ask = {
        "question": "How many times are failed payment requests retried and what happens afterwards?",
        "generate": not args.no_llm,
    }
    r = http.post("/api/v1/query", json=ask)
    check(r.status_code == 200, "query succeeded")
    body = r.json()
    top = body["sources"][0] if body["sources"] else {}
    check(top.get("filename") == "payment-runbook.md", "top source is the uploaded file")
    check(
        "exponential backoff" in top.get("excerpt", "") and "payments-dlq" in top.get("excerpt", ""),
        "source excerpt contains the evidence text",
    )
    check(top.get("section") == "Payment Service Runbook > Retry policy", "section metadata is correct")
    if not args.no_llm:
        answer = body["answer"] or ""
        check(not body["insufficient_evidence"], "model found sufficient evidence")
        check("4" in answer or "four" in answer.lower(), "answer states the retry count from the document")
        check(any(s["cited"] for s in body["sources"]), "answer cites at least one retrieved source")
    print("\nSmoke test passed. Now restart the server and re-run with --no-llm to verify persistence.")


if __name__ == "__main__":
    main()
