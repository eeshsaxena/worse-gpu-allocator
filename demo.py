"""Run a tiny, reproducible demo of the worse allocator."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.04,
        help="Seconds between rows, set to 0 for a fast run",
    )
    args = parser.parse_args()

    allocator = WorseGPUAllocator(
        gpu_capacity=32 * 1024 * 1024,
        gpu_probability=0.65,
        forget_probability=0.20,
        seed=args.seed,
    )
    live = []

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
        print(
            f"{step:>5}  {format_bytes(requested):>12}  "
            f"{allocation.device:>6}  "
            f"{format_bytes(allocation.reserved_bytes):>11}  "
            f"{format_bytes(stats['gpu_free']):>11}  "
            f"{stats['fragmentation']:>13.1%}"
        )
        time.sleep(max(0, args.delay))

    print("\nFinal state:")
    for key, value in allocator.snapshot().items():
        if key.endswith("bytes") or key.endswith("reserved") or key.endswith("used") or key.endswith("free") or key.endswith("forgotten"):
            value = format_bytes(value)
        print(f"  {key:>18}: {value}")


if __name__ == "__main__":
    main()
