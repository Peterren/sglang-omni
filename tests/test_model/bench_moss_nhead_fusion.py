#!/usr/bin/env python3
"""Micro-benchmark: MOSS-TTS N audio-head sampling — 32 sequential matmuls vs 1 fused.

Direction 5 (roadmap #1052) step: prove & size the per-frame host-side win of
fusing the uniform audio logit heads BEFORE touching the model code.

The MOSS-TTS decode path computes channel logits per frame as N separate heads
(`compute_channel_outputs` -> list-comp over `logits_processors`/`lm_heads`),
then immediately re-stacks the audio channels into one `[B, n_vq, V]` tensor
(`_sample_rows`, model_runner.py). Channel 0 is the huge text head (kept
separate); channels 1..n_vq are UNIFORM audio heads (vocab = audio_vocab+1,
padded) -> a clean single batched matmul.

This script times ONLY the audio-head portion (the fusible part):
  A (baseline): n_vq separate F.linear + torch.stack -> [B, n_vq, V]
  B (fused):    1 F.linear with [n_vq*V, D] weight -> reshape [B, n_vq, V]

Notes / honesty:
- Raw F.linear UNDER-counts the real win: production A also pays n_vq sglang
  LogitsProcessor Python invocations (metadata build, vocab slicing) per frame,
  which the fusion also removes. So B/A here is a conservative lower bound.
- Both methods produce numerically identical logits (same weights, same slice);
  --check asserts that.
- The text head (channel 0) is unchanged by the fusion, so it is excluded.

Applies to the NON-Local MOSS-TTS (`OpenMOSS-Team/MOSS-TTS-v1.5`), whose audio
heads are parallel/fusible. The Local variant uses an autoregressive local
transformer over codebooks and is NOT fusible this way.

Run on the GPU box (macOS/CPU has no meaningful launch-overhead signal):
  python tests/test_model/bench_moss_nhead_fusion.py \
      --model OpenMOSS-Team/MOSS-TTS-v1.5
  # or fully offline, pass shapes explicitly:
  python tests/test_model/bench_moss_nhead_fusion.py \
      --hidden-size 1024 --n-vq 32 --audio-vocab-padded 1088
"""

from __future__ import annotations

import argparse
import statistics

import torch


def _load_shapes_from_config(model: str) -> dict | None:
    """Best-effort: pull hidden_size / n_vq / audio vocab from the HF config."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    except Exception as exc:  # offline / no access -> caller falls back to args
        print(f"[config] could not load {model!r} ({exc}); using CLI shapes")
        return None

    def _get(obj, *names, default=None):
        for name in names:
            if hasattr(obj, name) and getattr(obj, name) is not None:
                return getattr(obj, name)
        return default

    hidden = int(_get(cfg, "hidden_size", default=0)) or int(
        _get(getattr(cfg, "language_config", cfg), "hidden_size", default=0)
    )
    n_vq = int(_get(cfg, "n_vq", default=32))
    audio_vocab = int(_get(cfg, "audio_vocab_size", default=1024))
    # ParallelLMHead pads vocab; the real head width is audio_vocab + 1, padded
    # up. We time at the PADDED width (that is the actual matmul size); default
    # pad to a multiple of 64 as sglang/TP typically does.
    real = audio_vocab + 1
    padded = ((real + 63) // 64) * 64
    if not hidden:
        return None
    return {"hidden_size": hidden, "n_vq": n_vq, "audio_vocab_padded": padded}


def _time_ms(fn, *, iters: int, warmup: int, device: str) -> float:
    """Return median per-call wall time in ms, CUDA-synced."""
    for _ in range(warmup):
        fn()
    if device == "cuda":
        torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        if device == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        else:
            import time

            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="OpenMOSS-Team/MOSS-TTS-v1.5")
    ap.add_argument("--hidden-size", type=int, default=0, help="override / offline")
    ap.add_argument("--n-vq", type=int, default=0, help="override / offline")
    ap.add_argument("--audio-vocab-padded", type=int, default=0, help="override")
    ap.add_argument("--batch-sizes", default="1,8,16")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--frame-ms", type=float, default=22.0,
                    help="nominal per-frame eager time (PR #751: ~22ms) to size impact")
    ap.add_argument("--check", action="store_true", help="assert A==B numerically")
    args = ap.parse_args()

    shapes = None
    if not (args.hidden_size and args.n_vq and args.audio_vocab_padded):
        shapes = _load_shapes_from_config(args.model)
    hidden = args.hidden_size or (shapes or {}).get("hidden_size")
    n_vq = args.n_vq or (shapes or {}).get("n_vq")
    vocab = args.audio_vocab_padded or (shapes or {}).get("audio_vocab_padded")
    if not (hidden and n_vq and vocab):
        ap.error("could not determine shapes; pass --hidden-size --n-vq --audio-vocab-padded")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    if device == "cpu":
        print("[warn] no CUDA — launch-overhead signal is meaningless on CPU")
        dtype = torch.float32
    else:
        print(f"[gpu] {torch.cuda.get_device_name(0)}")

    print(f"[shapes] hidden={hidden} n_vq={n_vq} audio_vocab_padded={vocab} "
          f"dtype={args.dtype} device={device}")

    torch.manual_seed(0)
    # n_vq separate audio head weights (baseline) and the fused equivalent.
    heads = [torch.randn(vocab, hidden, device=device, dtype=dtype) for _ in range(n_vq)]
    fused_w = torch.cat(heads, dim=0).contiguous()  # [n_vq*vocab, hidden]

    def make_a(h):
        def run():
            outs = [torch.nn.functional.linear(h, w) for w in heads]
            return torch.stack(outs, dim=1)  # [B, n_vq, vocab]
        return run

    def make_b(h):
        def run():
            logits = torch.nn.functional.linear(h, fused_w)  # [B, n_vq*vocab]
            return logits.reshape(h.shape[0], n_vq, vocab)
        return run

    if args.check:
        h = torch.randn(4, hidden, device=device, dtype=dtype)
        a, b = make_a(h)(), make_b(h)()
        max_err = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=1e-2, rtol=1e-2), f"A!=B, max_err={max_err}"
        print(f"[check] A == B (max abs err {max_err:.2e}) — fusion is numerically safe")

    print(f"\n{'batch':>6} {'A: 32 sep+stack':>18} {'B: 1 fused':>12} "
          f"{'speedup':>9} {'saved/frame':>12} {'% of 22ms':>10}")
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        h = torch.randn(bs, hidden, device=device, dtype=dtype)
        a_ms = _time_ms(make_a(h), iters=args.iters, warmup=args.warmup, device=device)
        b_ms = _time_ms(make_b(h), iters=args.iters, warmup=args.warmup, device=device)
        saved = a_ms - b_ms
        pct = 100.0 * saved / args.frame_ms
        print(f"{bs:>6} {a_ms:>16.4f}ms {b_ms:>10.4f}ms {a_ms / b_ms:>8.2f}x "
              f"{saved:>10.4f}ms {pct:>9.1f}%")

    print("\nReading: B/A is a conservative lower bound — production A also pays "
          f"{n_vq} sglang LogitsProcessor Python calls/frame the fusion removes.")


if __name__ == "__main__":
    main()
