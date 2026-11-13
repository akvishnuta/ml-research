"""
Rough memory estimate for running a model on your machine
"""
import argparse

GIB=1024**3 # 1024*3 bytes = 1 GB

def weights_gib(params_billion: float, bits_per_weight: float) -> float:
    """Size of the model wieght in GiB"""
    total_bits = params_billion * 1e9 * bits_per_weight
    return total_bits/8/GIB

def kv_cache_gib(layers: int, kv_heads: int, head_dim: int, context: int, bytes_per_value: int = 2) -> float:
    """Memory for the key and value cache ata a given context"""
    values = 2*layers* kv_heads * head_dim * context

    return values * bytes_per_value/GIB


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate memory needed to run a model."
    )
    parser.add_argument("params", type=float, help="parameters in billions, eg. 3.2")
    parser.add_argument("--bits", type=float, default=5.0, help="bits per weight after quantization")
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    args = parser.parse_args()

    if args.params <= 0 or args.bits <= 0 or args.context <= 0:
        parser.error("params, bits, and context must be positive")

    w = weights_gib(args.params, args.bits)
    kv = kv_cache_gib(args.layers, args.kv_heads, args.head_dim, args.context)

    print(f"Weights: {w:6.2f} GiB")

    print(f"Context cache: {kv:6.2f} GiB at {args.context} tokens")

    print(f"Total roughly: {w + kv:6.2f} GiB plus runtime overhead")


if __name__ == "__main__":
    main()

