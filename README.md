# vllm-private — Self-Hosted LLM in a Locked-Down VPC

Serve an open-weight LLM on **your own network only** — no public
endpoint, no data leaving the VPC — with vLLM on a GPU host, benchmarked
with GuideLLM-style metrics, and costed against a hosted API.

> **Honest status on this machine (25 Nov 2026):** this VM has **no GPU**
> (`nvidia-smi` not found, 4 CPU cores, 7.7 GB RAM). The **vLLM path was
> not run here** — vLLM needs a CUDA GPU and a multi-GB torch install.
> The demo below runs the **Ollama fallback path** instead: the same
> OpenAI-compatible API shape, the same benchmark harness, real measured
> numbers on CPU. vLLM numbers are **not** claimed anywhere in this repo.

## What it does

```
                  private network only (127.0.0.1 / VPC subnet)
  ┌─────────────┐   OpenAI-compatible    ┌──────────────────────┐
  │  client /   │  POST /v1/chat/       │  vLLM (GPU host)     │
  │  benchmark  │ ───────────────────▶  │  Qwen2.5-3B-Instruct │
  │  harness    │  Bearer token auth    │  + request logging   │
  └─────────────┘                       └──────────────────────┘
        │  Ollama fallback (no GPU): same API shape, same harness
        └────────────────────────────────▶ Ollama qwen2.5:3b-instruct
```

- **Serving** — `serve/` launches vLLM's OpenAI-compatible server bound
  to localhost/private IP only, with an API key and JSON request logging.
  Docker Compose and systemd variants included.
- **Benchmarking** — `bench/benchmark.py` measures the same metrics
  GuideLLM reports (time-to-first-token, output tokens/sec, request
  latency p50/p95/p99, throughput) against any OpenAI-compatible
  endpoint. A GuideLLM wrapper (`bench/guidellm_bench.py`) is included
  for GPU hosts where GuideLLM installs.
- **Cost comparison** — `cost/cost_model.py` computes cost per 1,000
  requests for self-hosted (GPU hourly rate) vs a hosted API
  (per-token price). **Every rate is a labelled assumption with a
  source and date — never a measured bill.**

## Quick start (no GPU needed — Ollama path)

```bash
./run.sh
```

That one command starts Ollama (if not already running), waits for the
`qwen2.5:3b-instruct` model, runs the benchmark, and prints the cost
comparison. Requires Ollama installed with the model pulled once:

```bash
ollama pull qwen2.5:3b-instruct   # ~1.9 GB, one time only
```

## vLLM path (GPU host)

Minimum GPU, honestly: a card with **≥ 8 GB VRAM** for Qwen2.5-3B-Instruct
in bf16 (~6 GB weights + KV cache; 12 GB+ recommended for comfortable
concurrency). A T4 (16 GB) is the practical floor; an L4 / A10G is
comfortable. No GPU → use the Ollama path above.

```bash
# Docker (recommended)
docker compose -f serve/docker-compose.yml up -d

# …or bare metal
bash serve/vllm_serve.sh

# …or systemd
sudo cp serve/vllm.service /etc/systemd/system/
sudo systemctl enable --now vllm
```

Then point the benchmark at it:

```bash
python3 bench/benchmark.py --base-url http://127.0.0.1:8000/v1 \
    --model Qwen/Qwen2.5-3B-Instruct --api-key "$VLLM_API_KEY"
```

## Architecture

| Piece | File | Notes |
|---|---|---|
| vLLM launch (bare metal) | `serve/vllm_serve.sh` | binds `127.0.0.1:8000`, `--api-key`, `--max-model-len 4096` |
| vLLM launch (Docker) | `serve/docker-compose.yml` | `vllm/vllm-openai`, GPU reservation, private network only |
| vLLM launch (systemd) | `serve/vllm.service` | restart-on-failure, env file for the API key |
| vLLM config | `serve/vllm_config.yaml` | model, limits, logging — one place to change |
| Benchmark harness | `bench/benchmark.py` | stdlib-only; TTFT, tok/s, p50/p95/p99, throughput |
| GuideLLM wrapper | `bench/guidellm_bench.py` | runs real GuideLLM when installed |
| Prompt set | `bench/prompts.jsonl` | 8 short prompts, fixed, so runs are comparable |
| Cost model | `cost/cost_model.py` | cost / 1K requests, self-hosted vs hosted API |
| Demo runner | `run.sh` | Ollama path end-to-end, one command |

## Real benchmark output (measured here, 25 Nov 2026)

Ollama path, `qwen2.5:3b-instruct` (Q4_K_M, 3.1B), CPU-only, 4 cores,
7.7 GB RAM, 8 prompts × 1 run, max 128 output tokens, sequential
(1 worker), wall time 74.0 s, 308 output tokens total (server-reported
usage). Full JSON:
[`results/benchmark-ollama-2026-11-25.json`](results/benchmark-ollama-2026-11-25.json).
Captured terminal output:
[`results/demo-terminal-output.txt`](results/demo-terminal-output.txt).

```
requests:            8/8 succeeded
time-to-first-token: mean 0.80 s | p50 0.39 s | p95 1.89 s
output tokens/sec:   mean 4.58 tok/s (per-request mean)
request latency:     p50 5.89 s | p95 24.21 s | mean 9.25 s
throughput:          6.49 req/min | 4.16 output tok/s overall
```

Run-to-run variance is real on this shared-CPU box: an earlier identical
run gave TTFT mean 1.39 s and 7.09 req/min. Treat these as
order-of-magnitude CPU numbers, not stable benchmarks.

These are **CPU numbers** — first-token latency is dominated by model
load on the first request (6.2 s) and prompt processing on CPU. A GPU
vLLM host would be an order of magnitude faster on TTFT and throughput;
that is the point of the comparison, not a claim. No vLLM numbers are
reported because none were run.

## Real cost comparison (assumptions, not bills)

From `./run.sh` output (defaults; see `cost/cost_model.py` for sources):

```
                          cost / 1,000 requests
self-hosted (T4 @ $0.35/hr assumed, 60 req/min):  $0.10
hosted API ($0.30 / 1M tokens assumed):            $0.03
```

At the throughput actually measured here on CPU (6.49 req/min — not a
GPU number, shown only to illustrate the volume effect), the same GPU
at $0.35/hr would be $0.90 / 1K requests: self-hosting only wins at
high sustained volume (break-even ≈ 202 req/min on these assumptions).

> Rates are **assumptions labelled with source + date** in
> `cost/cost_model.py` (e.g. public cloud GPU list prices, Nov 2026).
> They are not measured bills. Re-run with `--gpu-hourly` /
> `--hosted-per-1m` to plug in your real prices.

## Security model (locked-down VPC)

- Server binds `127.0.0.1` only (vLLM) / `127.0.0.1:11434` (Ollama).
  Nothing listens on a public interface.
- API key required (`--api-key` / `VLLM_API_KEY` env file, never in git).
- No secrets anywhere in this repo — no API keys, tokens, or passwords.
- In a real VPC: put the server in a **private subnet with no public
  IP**, allow inbound 8000 only from the app subnet's security group,
  egress only to the model-weights bucket at deploy time, and ship logs
  to the VPC's log store. TLS terminates at the internal load balancer.
- Request logging is on (`--log-requests` equivalent in
  `serve/vllm_config.yaml`); prompts are logged only if you enable it —
  keep prompt logging off for user data.

## GPU requirements & limitations (honest)

- **This VM: no GPU.** `nvidia-smi: command not found`, no torch
  installed. vLLM 0.34.0 exists on PyPI (checked 25 Nov 2026) but was
  **not installed here** — it pulls a multi-GB CUDA torch and cannot
  run without a GPU anyway.
- **GuideLLM:** install was attempted in a venv on this VM
  (25 Nov 2026); see `bench/README.md` for the outcome. The bundled
  harness mirrors GuideLLM's methodology (streaming TTFT, per-request
  output tok/s, latency percentiles, sequential + concurrent workers)
  and is what produced the numbers above.
- **Ollama ≠ vLLM performance.** CPU Ollama numbers say nothing about
  vLLM-on-GPU throughput. The harness is portable; the numbers are not.
- **Model:** Qwen2.5-3B-Instruct (Apache 2.0, open-weight). 3B is a
  demo-size model — fine for latency/cost methodology, not a quality
  benchmark.
- **Cost model:** assumes a fixed prompt/output token mix (see
  `cost/cost_model.py`); your real mix changes the answer.

## Files

```
vllm-private/
├── run.sh                    # one-command demo (Ollama path)
├── serve/
│   ├── vllm_serve.sh         # bare-metal vLLM launch
│   ├── docker-compose.yml    # vLLM in Docker, GPU + private network
│   ├── vllm.service          # systemd unit
│   └── vllm_config.yaml      # model + serving config
├── bench/
│   ├── benchmark.py          # GuideLLM-style harness (stdlib only)
│   ├── guidellm_bench.py     # real GuideLLM wrapper (GPU hosts)
│   ├── prompts.jsonl         # fixed prompt set
│   └── README.md             # GuideLLM install outcome + methodology
├── cost/
│   └── cost_model.py         # cost / 1K requests comparison
├── results/
│   ├── benchmark-ollama-2026-11-25.json
│   └── demo-terminal-output.txt
└── requirements.txt          # requests only (benchmark + cost)
```
