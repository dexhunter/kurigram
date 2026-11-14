"""Runtime of the `xor` calls inside one 1 MiB AES-256-IGE file-chunk decrypt.

`ige` xors twice per 16-byte block: once before `pyaes` and once after. A
1 MiB `upload.GetFile` chunk is 65536 blocks, so one decrypt performs
131072 xors of 16-byte strings. Pairs are built before the timer. The timed
section only calls `xor`.
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

from pyrogram.utils import xor

# Two 16-byte xors per block of the 1 MiB chunk `Client.get_file` requests.
BLOCK = 16
CHUNK = 1024 * 1024
CALLS = (CHUNK // BLOCK) * 2
WARMUP = 2
SAMPLES = 7


def _pairs() -> list[tuple[bytes, bytes]]:
    rng = random.Random(0)
    blob = rng.randbytes(CHUNK)
    pairs: list[tuple[bytes, bytes]] = []
    for offset in range(0, CHUNK, BLOCK):
        block = blob[offset : offset + BLOCK]
        pairs.append((block, rng.randbytes(BLOCK)))
        pairs.append((rng.randbytes(BLOCK), block))
    if len(pairs) != CALLS:
        raise SystemExit(f"expected {CALLS} xors, built {len(pairs)}")
    # The same bytes `ige` feeds `xor`: equal lengths, 16 bytes each.
    sample = xor(pairs[0][0], pairs[0][1])
    if len(sample) != BLOCK or xor(pairs[0][1], pairs[0][0]) != sample:
        raise SystemExit("xor changed a 16-byte block")
    return pairs


def main() -> None:
    pairs = _pairs()
    gc.collect()
    gc.disable()
    try:
        for _ in range(WARMUP):
            acc = 0
            for left, right in pairs:
                acc ^= xor(left, right)[0]
            if acc < 0:
                raise SystemExit("xor accumulator went negative")

        samples: list[float] = []
        for _ in range(SAMPLES):
            started = time.perf_counter()
            acc = 0
            for left, right in pairs:
                acc ^= xor(left, right)[0]
            samples.append((time.perf_counter() - started) * 1000.0)
            if acc < 0:
                raise SystemExit("xor accumulator went negative")
    finally:
        gc.enable()

    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    print(f"median_ms={median:.4f}")
    print(f"noise_pct={noise:.4f}")
    print("samples=" + ",".join(f"{sample:.4f}" for sample in samples))


if __name__ == "__main__":
    main()
