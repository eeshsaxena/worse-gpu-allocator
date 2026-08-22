import random
from pathlib import Path
import subprocess
import sys
import unittest

from benchmark import ReferenceGPUAllocator
from worse_gpu_allocator import WorseGPUAllocator


ROOT = Path(__file__).parent


class RobustnessTests(unittest.TestCase):
    def test_randomized_workloads_preserve_invariants(self) -> None:
        for seed in range(20):
            with self.subTest(seed=seed):
                chooser = random.Random(seed)
                allocator = WorseGPUAllocator(
                    gpu_capacity=1 << 20,
                    gpu_probability=0.55,
                    forget_probability=0,
                    trace_limit=32,
                    seed=seed,
                )
                live = []
                for _ in range(200):
                    if live and chooser.random() < 0.45:
                        allocator.free(live.pop(chooser.randrange(len(live))))
                    else:
                        live.append(allocator.allocate(chooser.randint(1, 32_768)))
                    allocator.validate_invariants()

                for allocation in live:
                    allocator.free(allocation)
                allocator.validate_invariants()
                stats = allocator.snapshot()
                self.assertEqual(stats["active_allocations"], 0)
                self.assertEqual(stats["cpu_bytes"], 0)
                self.assertEqual(stats["gpu_used"], 0)
                self.assertGreaterEqual(stats["gpu_unreserved"], 0)

    def test_strict_oom_is_atomic(self) -> None:
        allocator = WorseGPUAllocator(
            gpu_capacity=1,
            fallback_on_oom=False,
            gpu_probability=1,
            seed=1,
        )
        before = allocator.snapshot()
        with self.assertRaises(MemoryError):
            allocator.allocate(1024)
        self.assertEqual(allocator.snapshot(), before)
        allocator.validate_invariants()

    def test_reference_cross_free_does_not_corrupt_state(self) -> None:
        first = ReferenceGPUAllocator(1024)
        second = ReferenceGPUAllocator(1024)
        handle = first.allocate(128)
        with self.assertRaises(KeyError):
            second.free(handle)
        self.assertEqual(first.snapshot()["active_allocations"], 1)
        first.free(handle)
        self.assertEqual(first.snapshot()["active_allocations"], 0)

    def test_cli_rejects_invalid_workload_sizes(self) -> None:
        for script in ("demo.py", "benchmark.py"):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, script, "--steps", "0"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()

