"""Run a tiny, reproducible demo of the worse allocator."""

from __future__ import annotations

import argparse
import json
import time

from worse_gpu_allocator import WorseGPUAllocator


def format_bytes(value: int | float) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    number = float(value)
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:,.1f} {unit}"
        number /= 1024
    return f"{number:,.1f} GiB"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or positive")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=_positive_int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--delay",
        type=_non_negative_float,
        default=0.04,
        help="Seconds between rows, set to 0 for a fast run",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only the final metrics as JSON",
    )
    args = parser.parse_args()

    allocator = WorseGPUAllocator(
        gpu_capacity=32 * 1024 * 1024,
        gpu_probability=0.65,
        forget_probability=0.20,
        seed=args.seed,
    )
    live = []

    if not args.json:
        print("A Worse GPU Memory Allocator")
        print("It allocates memory on the GPU (sometimes).\n")
        print(" step  request       device  reserved     gpu free     fragmentation")
        print("-----  ------------  ------  -----------  -----------  -------------")

    for step in range(1, args.steps + 1):
        requested = (step * 7919) % (2 * 1024 * 1024) + 4096
        allocation = allocator.allocate(requested)
        live.append(allocation)

        if len(live) > 3 and step % 3 == 0:
            old = live.pop(0)
            allocator.free(old)

        stats = allocator.snapshot()
        if not args.json:
            print(
                f"{step:>5}  {format_bytes(requested):>12}  "
                f"{allocation.device:>6}  "
                f"{format_bytes(allocation.reserved_bytes):>11}  "
                f"{format_bytes(stats['gpu_free']):>11}  "
                f"{stats['fragmentation']:>13.1%}"
            )
        time.sleep(max(0, args.delay))

    final_stats = allocator.snapshot()
    if args.json:
        print(json.dumps(final_stats, sort_keys=True))
    else:
        print("\nFinal state:")
        for key, value in final_stats.items():
            if key.endswith("bytes") or key.endswith("reserved") or key.endswith("used") or key.endswith("free") or key.endswith("forgotten") or key.endswith("unreserved"):
                value = format_bytes(value)
            print(f"  {key:>18}: {value}")


if __name__ == "__main__":
    main()
