#!/usr/bin/env python3
"""GuideLLM-style benchmark harness (stdlib only).

Measures the same metrics GuideLLM reports, against any OpenAI-compatible
/chat/completions endpoint (vLLM or Ollama):

  * time-to-first-token (TTFT) — streaming: request sent -> first token chunk
  * output tokens/sec — completion tokens / generation time
  * request latency — total time per request (p50 / p95 / p99 / mean)
  * throughput — completed requests / minute (and tokens/sec overall)

Methodology notes (mirrors GuideLLM):
  * Streaming is used so TTFT is a real first-token measurement, not
    end-of-response. A non-streaming fallback is used if the server
    refuses `stream: true`.
  * Token counts come from the server's `usage` block when present
    (vLLM and Ollama both return it); otherwise completion tokens are
    estimated as len(text.split()) and flagged as estimated.
  * Prompts are fixed (prompts.jsonl next to this file) so runs are
    comparable. Concurrency is configurable; the default is sequential
    (1 worker), which is the honest baseline on a CPU box.

Usage:
  python3 bench/benchmark.py [--base-url URL] [--model MODEL]
      [--api-key KEY] [--prompts FILE] [--max-tokens N]
      [--concurrency N] [--output FILE]
"""
import argparse
import json
import statistics
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _post(url, payload, api_key, timeout=300):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    return urllib.request.urlopen(req, timeout=timeout)


def run_one(base_url, model, prompt, max_tokens, api_key):
    """One streamed chat completion. Returns a dict of measurements."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        # vLLM honours this and returns usage in the final chunk;
        # Ollama ignores it (no usage in streaming), handled below.
        "stream_options": {"include_usage": True},
    }
    t0 = time.monotonic()
    first_token_at = None
    chunks = []
    usage = None
    try:
        with _post(url, payload, api_key) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if first_token_at is None:
                    first_token_at = time.monotonic()
                if obj.get("usage"):
                    usage = obj["usage"]
                choice = (obj.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    chunks.append(delta["content"])
    except urllib.error.HTTPError as e:
        # Fallback: server may not support streaming; retry non-streaming.
        if e.code in (400, 422):
            payload["stream"] = False
            t0 = time.monotonic()
            with _post(url, payload, api_key) as resp:
                obj = json.loads(resp.read())
            text = (obj.get("choices") or [{}])[0].get("message", {}).get("content", "")
            usage = obj.get("usage")
            t1 = time.monotonic()
            return {
                "ok": True, "streamed": False, "ttft_s": None,
                "latency_s": t1 - t0, "text": text, "usage": usage,
                "estimated_tokens": len(text.split()),
            }
        raise
    t1 = time.monotonic()
    text = "".join(chunks)
    return {
        "ok": True, "streamed": True,
        "ttft_s": (first_token_at - t0) if first_token_at is not None else None,
        "latency_s": t1 - t0, "text": text, "usage": usage,
        # Ollama streams one chunk per token and returns no usage block,
        # so chunk count is the token estimate there (labelled estimated).
        "estimated_tokens": len(chunks),
    }


def percentile(values, pct):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def main():
    here = Path(__file__).parent
    ap = argparse.ArgumentParser(description="GuideLLM-style benchmark harness")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model", default="qwen2.5:3b-instruct")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--prompts", default=str(here / "prompts.jsonl"))
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    prompts = [json.loads(l)["prompt"] for l in open(args.prompts)
               if l.strip()]
    print(f"endpoint : {args.base_url}  model: {args.model}")
    print(f"prompts  : {len(prompts)}  max_tokens: {args.max_tokens}  "
          f"concurrency: {args.concurrency}")

    def job(p):
        try:
            r = run_one(args.base_url, args.model, p, args.max_tokens,
                        args.api_key)
            return {"prompt": p, **r}
        except Exception as e:  # noqa: BLE001 - report per-request failure
            return {"prompt": p, "ok": False, "error": str(e)}

    t_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        results = list(ex.map(job, prompts))
    wall = time.monotonic() - t_start

    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]
    for r in failed:
        print(f"FAILED: {r['prompt'][:60]!r} -> {r.get('error')}", file=sys.stderr)

    ttfts = [r["ttft_s"] for r in ok if r.get("ttft_s") is not None]
    lats = [r["latency_s"] for r in ok]

    def toks(r):
        if r.get("usage") and r["usage"].get("completion_tokens") is not None:
            return r["usage"]["completion_tokens"], False
        return r.get("estimated_tokens", 0), True

    tok_rates, total_out = [], 0
    estimated = False
    for r in ok:
        n, est = toks(r)
        estimated |= est
        total_out += n
        gen = r["latency_s"] - (r["ttft_s"] or 0)
        if gen > 0:
            tok_rates.append(n / gen)

    summary = {
        "endpoint": args.base_url,
        "model": args.model,
        "n_requests": len(prompts),
        "n_succeeded": len(ok),
        "n_failed": len(failed),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "wall_s": round(wall, 2),
        "ttft_s": {
            "mean": round(statistics.mean(ttfts), 3) if ttfts else None,
            "p50": round(percentile(ttfts, 50), 3) if ttfts else None,
            "p95": round(percentile(ttfts, 95), 3) if ttfts else None,
        },
        "output_tokens_per_s_per_request": {
            "mean": round(statistics.mean(tok_rates), 2) if tok_rates else None,
        },
        "latency_s": {
            "mean": round(statistics.mean(lats), 3) if lats else None,
            "p50": round(percentile(lats, 50), 3) if lats else None,
            "p95": round(percentile(lats, 95), 3) if lats else None,
            "p99": round(percentile(lats, 99), 3) if lats else None,
        },
        "throughput_req_per_min": round(len(ok) / wall * 60, 2) if wall else None,
        "throughput_output_tokens_per_s": round(total_out / wall, 2) if wall else None,
        "total_output_tokens": total_out,
        "tokens_estimated": estimated,
        "streamed": all(r.get("streamed") for r in ok) if ok else False,
    }

    print("\n--- benchmark results ---")
    print(f"requests            : {len(ok)}/{len(prompts)} succeeded")
    if ttfts:
        print(f"time-to-first-token : mean {summary['ttft_s']['mean']} s | "
              f"p50 {summary['ttft_s']['p50']} s | p95 {summary['ttft_s']['p95']} s")
    print(f"output tokens/sec   : mean {summary['output_tokens_per_s_per_request']['mean']} tok/s (per-request mean)")
    print(f"request latency     : p50 {summary['latency_s']['p50']} s | "
          f"p95 {summary['latency_s']['p95']} s | mean {summary['latency_s']['mean']} s")
    print(f"throughput          : {summary['throughput_req_per_min']} req/min | "
          f"{summary['throughput_output_tokens_per_s']} output tok/s overall")
    if estimated:
        print("note: token counts estimated from streamed chunks "
              "(server gave no usage block; Ollama streams ~1 chunk/token)")

    out = args.output
    if out:
        Path(out).write_text(json.dumps(
            {"summary": summary, "requests": results}, indent=2))
        print(f"saved: {out}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
