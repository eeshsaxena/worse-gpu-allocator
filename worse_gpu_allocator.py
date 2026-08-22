"""A deliberately bad GPU memory allocator.

This is a small, dependency-free simulator intended as a teaching/demo project.
It models the sort of allocator you should not ship:

* allocations go to the GPU only with a probability;
* requests are padded into awkwardly sized blocks;
* freed blocks are not coalesced, so fragmentation grows;
* some frees are "forgotten" and remain reserved;
* allocation performs a deliberately slow linear scan.

The allocator never touches a real GPU. That makes the joke reproducible on any
machine and keeps the project useful in tests and demos.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Literal


__version__ = "0.1.0"
__all__ = ["Allocation", "OutOfMemoryError", "WorseGPUAllocator", "__version__"]


Device = Literal["gpu", "cpu"]


@dataclass(frozen=True)
class Allocation:
    """A handle returned by :meth:`WorseGPUAllocator.allocate`."""

    allocation_id: int
    requested_bytes: int
    reserved_bytes: int
    device: Device
    block_index: int


@dataclass
class Block:
    """One reserved block in the fake GPU arena."""

    size: int
    allocation_id: int | None = None
    forgotten: bool = False

    @property
    def free(self) -> bool:
        return self.allocation_id is None and not self.forgotten


class OutOfMemoryError(MemoryError):
    """Raised when the fake GPU cannot satisfy a request."""


class WorseGPUAllocator:
    """An allocator whose design goals are inconsistency and regret.

    Args:
        gpu_capacity: Simulated GPU capacity in bytes.
        gpu_probability: Chance that an allocation actually uses the GPU.
        forget_probability: Chance that ``free`` forgets to release a GPU block.
        seed: Seed for reproducible bad decisions.
    """

    def __init__(
        self,
        gpu_capacity: int = 64 * 1024 * 1024,
        gpu_probability: float = 0.65,
        forget_probability: float = 0.20,
        fallback_on_oom: bool = True,
        seed: int | None = None,
    ) -> None:
        if gpu_capacity <= 0:
            raise ValueError("gpu_capacity must be positive")
        if not 0 <= gpu_probability <= 1:
            raise ValueError("gpu_probability must be between 0 and 1")
        if not 0 <= forget_probability <= 1:
            raise ValueError("forget_probability must be between 0 and 1")

        self.gpu_capacity = gpu_capacity
        self.gpu_probability = gpu_probability
        self.forget_probability = forget_probability
        self.fallback_on_oom = fallback_on_oom
        self._random = random.Random(seed)
        self._blocks: List[Block] = []
        self._allocations: Dict[int, Allocation] = {}
        self._next_id = 1
        self.cpu_bytes = 0

    def allocate(self, requested_bytes: int) -> Allocation:
        """Reserve memory, occasionally on the GPU, and return its handle."""

        if requested_bytes <= 0:
            raise ValueError("requested_bytes must be positive")

        if self._random.random() >= self.gpu_probability:
            allocation = Allocation(
                allocation_id=self._next_id,
                requested_bytes=requested_bytes,
                reserved_bytes=requested_bytes,
                device="cpu",
                block_index=-1,
            )
            self._next_id += 1
            self._allocations[allocation.allocation_id] = allocation
            self.cpu_bytes += requested_bytes
            return allocation

        reserved_bytes = self._bad_rounding(requested_bytes)
        block_index = self._find_first_fit(reserved_bytes)
        if block_index is None:
            if not self.fallback_on_oom:
                raise OutOfMemoryError(
                    f"GPU request for {requested_bytes} bytes could not be reserved"
                )
            # A real allocator might compact or retry. This one gives up and
            # quietly sends the work to CPU, which is worse in a different way.
            allocation = Allocation(
                allocation_id=self._next_id,
                requested_bytes=requested_bytes,
                reserved_bytes=requested_bytes,
                device="cpu",
                block_index=-1,
            )
            self._next_id += 1
            self._allocations[allocation.allocation_id] = allocation
            self.cpu_bytes += requested_bytes
            return allocation

        allocation = Allocation(
            allocation_id=self._next_id,
            requested_bytes=requested_bytes,
            reserved_bytes=reserved_bytes,
            device="gpu",
            block_index=block_index,
        )
        self._next_id += 1
        self._blocks[block_index].allocation_id = allocation.allocation_id
        self._allocations[allocation.allocation_id] = allocation
        return allocation

    def free(self, allocation: Allocation) -> bool:
        """Free an allocation, unless the allocator randomly forgets.

        Returns ``True`` when the memory was actually released and ``False``
        when it became a forgotten reservation.
        """

        current = self._allocations.pop(allocation.allocation_id, None)
        if current is None:
            raise KeyError(f"unknown allocation {allocation.allocation_id}")

        if current.device == "cpu":
            self.cpu_bytes -= current.reserved_bytes
            return True

        block = self._blocks[current.block_index]
        if self._random.random() < self.forget_probability:
            block.forgotten = True
            return False

        block.allocation_id = None
        return True

    def snapshot(self) -> dict[str, int | float]:
        """Return simple metrics suitable for a CLI or dashboard."""

        reserved = sum(block.size for block in self._blocks)
        used = sum(
            block.size
            for block in self._blocks
            if block.allocation_id is not None
        )
        free = sum(block.size for block in self._blocks if block.free)
        forgotten = sum(block.size for block in self._blocks if block.forgotten)
        return {
            "gpu_capacity": self.gpu_capacity,
            "gpu_reserved": reserved,
            "gpu_used": used,
            "gpu_free": free,
            "gpu_forgotten": forgotten,
            "gpu_blocks": len(self._blocks),
            "cpu_bytes": self.cpu_bytes,
            "active_allocations": len(self._allocations),
            "fragmentation": round(1 - (free / reserved), 3) if reserved else 0.0,
        }

    def _find_first_fit(self, size: int) -> int | None:
        # Intentional O(n) scan. It also refuses to split a larger block,
        # creating needless waste and more fragmentation.
        for index, block in enumerate(self._blocks):
            if block.free and block.size == size:
                return index

        reserved = sum(block.size for block in self._blocks)
        if reserved + size > self.gpu_capacity:
            return None

        self._blocks.append(Block(size=size))
        return len(self._blocks) - 1

    def _bad_rounding(self, size: int) -> int:
        """Round to a random bucket, occasionally wasting almost 2x memory."""

        bucket = self._random.choice((256, 1024, 4096, 16384))
        rounded = ((size + bucket - 1) // bucket) * bucket
        if self._random.random() < 0.12:
            rounded *= 2
        return rounded
