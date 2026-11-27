#!/usr/bin/env python3
"""Cost per 1,000 requests: self-hosted (vLLM on a rented GPU) vs hosted API.

Every rate below is a LABELLED ASSUMPTION with a source and date — not a
measured bill. Change them with the CLI flags to plug in your real prices.

Defaults (assumed, Nov 2026):
  * Self-hosted GPU: T4 16 GB at $0.35/hr — public list-price range for a
    T4 on demand cloud GPU marketplaces (e.g. ~$0.35–$0.60/hr, Nov 2026).
    Picked the low end; a larger GPU (L4/A10G) costs more per hour.
  * Hosted API: $0.30 per 1M tokens blended (input+output) — in the range
    of small-model hosted pricing for ~3B-class models (e.g. $0.10–$0.50
    per 1M tokens, Nov 2026). This is a round illustrative number, not a
    vendor quote.
  * Token mix per request (measured from this repo's benchmark prompts):
    ~36 prompt tokens in, ~60 output tokens out on average. The script
    accepts --prompt-tokens / --output-tokens to override.

Usage:
  python3 cost/cost_model.py [--gpu-hourly 0.35] [--hosted-per-1m 0.30]
      [--req-per-min 60] [--prompt-tokens 36] [--output-tokens 60]
"""
import argparse

ASSUMPTIONS = {
    "gpu_hourly_usd": (0.35, "Assumed T4 16GB on-demand list price, "
                             "public cloud GPU marketplaces, Nov 2026 "
                             "(range seen ~$0.35-0.60/hr)"),
    "hosted_per_1m_usd": (0.30, "Assumed blended $/1M tokens for a ~3B-class "
                                "hosted model, Nov 2026 (range ~$0.10-0.50)"),
}


def main():
    ap = argparse.ArgumentParser(description="Cost per 1K requests comparison")
    ap.add_argument("--gpu-hourly", type=float, default=ASSUMPTIONS["gpu_hourly_usd"][0])
    ap.add_argument("--hosted-per-1m", type=float, default=ASSUMPTIONS["hosted_per_1m_usd"][0])
    ap.add_argument("--req-per-min", type=float, default=60,
                    help="sustained throughput of the self-hosted server")
    ap.add_argument("--prompt-tokens", type=float, default=36)
    ap.add_argument("--output-tokens", type=float, default=60)
    ap.add_argument("--utilization", type=float, default=1.0,
                    help="fraction of the hour the GPU is actually serving (1.0 = fully busy)")
    args = ap.parse_args()

    req_per_hour = args.req_per_min * 60 * args.utilization
    self_per_1k = args.gpu_hourly / req_per_hour * 1000 if req_per_hour else float("inf")
    hosted_per_req = (args.prompt_tokens + args.output_tokens) / 1_000_000 * args.hosted_per_1m
    hosted_per_1k = hosted_per_req * 1000

    print("cost per 1,000 requests  (ASSUMED rates — not measured bills)")
    print("=" * 62)
    print(f"self-hosted vLLM (GPU ${args.gpu_hourly:.2f}/hr, "
          f"{args.req_per_min:.0f} req/min, {args.utilization:.0%} utilized):")
    print(f"  ${self_per_1k:>8.2f} / 1K requests")
    print(f"hosted API (${args.hosted_per_1m:.2f}/1M tokens, "
          f"{args.prompt_tokens:.0f} in + {args.output_tokens:.0f} out tokens/req):")
    print(f"  ${hosted_per_1k:>8.2f} / 1K requests")
    print("-" * 62)
    if hosted_per_1k > 0:
        ratio = self_per_1k / hosted_per_1k
        cheaper = "self-hosted" if self_per_1k < hosted_per_1k else "hosted API"
        print(f"cheaper at this volume: {cheaper} "
              f"(self-hosted is {ratio:.2f}x the hosted cost)")
    print(f"break-even throughput: hosted cost = GPU cost at "
          f"{args.gpu_hourly / hosted_per_req / 60:.1f} req/min "
          f"(above this, self-hosted wins on these assumptions)")
    print()
    print("assumptions (sources, Nov 2026):")
    print(f"  GPU ${args.gpu_hourly:.2f}/hr — {ASSUMPTIONS['gpu_hourly_usd'][1]}")
    print(f"  hosted ${args.hosted_per_1m:.2f}/1M tokens — {ASSUMPTIONS['hosted_per_1m_usd'][1]}")
    print("  token mix from this repo's benchmark prompts; "
          "re-run with --prompt-tokens/--output-tokens for your workload.")
    print("  GPU cost is per hour whether busy or idle; at low utilization")
    print("  (e.g. 10% busy) the self-hosted cost per request is 10x higher.")


if __name__ == "__main__":
    main()
