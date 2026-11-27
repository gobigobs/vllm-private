# Benchmarking — GuideLLM status + methodology

## GuideLLM install outcome (this VM, 25 Nov 2026)

`pip install guidellm` was attempted twice in a repo venv on 25 Nov 2026
and **did not install here**:

1. First attempt: failed with `OSError: [Errno 28] No space left on device`
   while downloading the torch 2.15.0 wheel (539 MB) — pip's temp dir
   defaults to `/tmp`, a 512 MB tmpfs on this VM.
2. Second attempt (with `TMPDIR` pointed at the repo and
   `--no-cache-dir`): still building after **21 minutes** on this
   1.5-shared-CPU VM (2.3 GB of wheels unpacked, torch still not
   installed) and was stopped. GuideLLM pulls torch + a large
   dependency tree; it is not a lightweight install.

So **no GuideLLM numbers exist for this repo** — the wrapper
(`bench/guidellm_bench.py`) refuses honestly instead of faking them, and
the numbers in the README come from `bench/benchmark.py` below. On a
real GPU host, `pip install guidellm` + `bench/guidellm_bench.py` is the
way to get true GuideLLM numbers.

## What the bundled harness measures

`bench/benchmark.py` (stdlib only, no install needed) mirrors GuideLLM's
core methodology for a fixed prompt set:

| GuideLLM metric | This harness |
|---|---|
| Time to first token (TTFT) | streaming: request sent → first `data:` chunk with content |
| Output tokens / sec | `usage.completion_tokens` / (latency − TTFT), per request |
| Request latency | total time per request; p50 / p95 / p99 / mean |
| Throughput | completed requests / minute + output tok/s overall |
| Concurrency | `--concurrency N` (default 1 = sequential baseline) |

Differences from real GuideLLM: no Poisson request-rate sweep, no
per-token latency distribution, no built-in dataset generators — the
prompt set is the fixed `prompts.jsonl` in this folder so runs are
comparable. For a GPU host, run the real thing via
`bench/guidellm_bench.py`.
