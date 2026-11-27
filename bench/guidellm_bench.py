#!/usr/bin/env python3
"""Run a real GuideLLM benchmark against the local endpoint — GPU hosts only.

GuideLLM (https://github.com/vllm-project/guidellm) is the standard
benchmark for vLLM servers. This wrapper shells out to its CLI so the
numbers it prints are GuideLLM's own, not this repo's harness.

Install (GPU host, in a venv):
    pip install guidellm

Run:
    python3 bench/guidellm_bench.py --base-url http://127.0.0.1:8000/v1 \
        --model Qwen/Qwen2.5-3B-Instruct --api-key "$VLLM_API_KEY"

If GuideLLM is not installed, this script says so and points at
bench/benchmark.py, the stdlib harness that mirrors GuideLLM's metrics
(TTFT, output tok/s, latency percentiles, throughput). It never fabricates
GuideLLM numbers.
"""
import argparse
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description="GuideLLM benchmark wrapper")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--rate", type=float, default=1.0, help="requests/sec")
    ap.add_argument("--num-prompts", type=int, default=20)
    ap.add_argument("--output", default="results/guidellm-benchmark.json")
    args = ap.parse_args()

    if shutil.which("guidellm") is None:
        print("GuideLLM is NOT installed on this machine.", file=sys.stderr)
        print("Install it with:  pip install guidellm", file=sys.stderr)
        print("Or run the equivalent stdlib harness instead:", file=sys.stderr)
        print("  python3 bench/benchmark.py --base-url "
              f"{args.base_url} --model {args.model}", file=sys.stderr)
        return 2

    cmd = [
        "guidellm", "benchmark",
        "--target", args.base_url,
        "--model", args.model,
        "--data", "prompt_tokens=128,output_tokens=128",
        "--rate", str(args.rate),
        "--num-prompts", str(args.num_prompts),
        "--output-path", args.output,
    ]
    env = None
    if args.api_key:
        import os
        env = dict(os.environ, OPENAI_API_KEY=args.api_key)
    print("running:", " ".join(cmd))
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
