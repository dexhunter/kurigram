"""Median runtime of pyrogram.raw.core.primitives.bytes.Bytes.read.

Inputs are TL-framed strings and file references: short usernames and mime
types, captions, and message bodies on both sides of the 253-byte length
cutoff. Each buffer is rewound and then read. The timed section calls
Bytes.read and BytesIO.seek.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import random
import statistics
import sys
import time
from io import BytesIO
from pathlib import Path


def _stabilize_process() -> None:
    """Re-exec once with ASLR off so repeated runs share an allocator layout."""
    if os.environ.get("OSSPR_BENCH_STABLE") == "1":
        try:
            os.sched_setaffinity(0, {0})
        except (AttributeError, OSError):
            pass
        return

    if ctypes.CDLL(None).personality(0x0040000) >= 0:
        os.environ["PYTHONHASHSEED"] = "0"
        os.environ["OSSPR_BENCH_STABLE"] = "1"
        os.execv(sys.executable, [sys.executable, *sys.argv])


_stabilize_process()
logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from pyrogram.raw.core.primitives.bytes import Bytes

LENGTHS: tuple[int, ...] = (
    *([5, 8, 12, 16, 24, 32, 48] * 12),
    *([20, 36, 64, 80, 128, 180, 220] * 4),
    200,
    253,
    254,
    300,
    400,
    512,
)

COPIES = 1600
WARMUP = 2
SAMPLES = 11


def corpus() -> list[BytesIO]:
    rng = random.Random(0)
    buffers: list[BytesIO] = []
    for length in LENGTHS:
        value = rng.randbytes(length)
        framed = bytes(Bytes(value))
        if Bytes.read(BytesIO(framed)) != value:
            raise SystemExit("Bytes.read did not round-trip")
        buffers.append(BytesIO(framed))
    return buffers * COPIES


def main() -> None:
    buffers = corpus()
    gc.collect()
    gc.disable()
    try:
        for _ in range(WARMUP):
            for buffer in buffers:
                buffer.seek(0)
                Bytes.read(buffer)

        samples: list[float] = []
        for _ in range(SAMPLES):
            started = time.perf_counter()
            consumed = 0
            for buffer in buffers:
                buffer.seek(0)
                consumed += len(Bytes.read(buffer))
            samples.append((time.perf_counter() - started) * 1000.0)
            if consumed <= 0:
                raise SystemExit("Bytes.read produced no output")
    finally:
        gc.enable()

    print(f"median_ms={statistics.median(samples):.4f}")


if __name__ == "__main__":
    main()
