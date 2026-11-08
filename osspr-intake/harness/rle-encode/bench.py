"""Median runtime of pyrogram.file_id.rle_encode on real file-id payloads.

The inputs are the raw bodies of file ids and file unique ids from the unit
tests: exactly the bytes FileId.encode and FileUniqueId.encode pass to
rle_encode. The timed section only calls rle_encode.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
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

from pyrogram.file_id import b64_decode, rle_decode, rle_encode

# File ids and file unique ids that tests/unit/test_file_id.py round-trips.
FILE_IDS: tuple[str, ...] = (
    "CQACAgIAAx0CAAGgr9AAAgmQX7b4XPBstC1fFUuJBooHTHFd7HMAAgUAA4GkuUnVOGG5P196yR4E",
    "AgADBQADgaS5SQ",
    "AAMCAgADHQIAAaCv0AACCZBftvhc8Gy0LV8VS4kGigdMcV3scwACBQADgaS5SdU4Ybk_"
    "X3rJIH3qihAAAwEAB20AA_OeAQABHgQ",
    "AQADIH3qihAAA_OeAQAB",
    "BAACAgIAAx0CAAGgr9AAAgmRX7b4Xv9f-4BK5VR_5ppIOF6UIp0AAgYAA4GkuUmhnZz2xC37wR4E",
    "AgADBgADgaS5SQ",
    "BQACAgIAAx0CAAGgr9AAAgmPX7b4UxbjNoFEO_L0I4s6wrXNJA8AAgQAA4GkuUm9FFvIaOhXWR4E",
    "AgADBAADgaS5SQ",
    "CgACAgIAAx0CAAGgr9AAAgmSX7b4Y2g8_QW2XFd49iUmRnHOyG8AAgcAA4GkuUnry9gWDzF_5R4E",
    "AwACAgIAAx0CAAGgr9AAAgmUX7b4c1KQyHVwzffxC2EnSYWsMAQAAgkAA4GkuUlsZUZ4_I97AR4E",
    "DQACAgIAAx0CAAGgr9AAAgmVX7b53qrRzCEO13BaLQJaYuFbdlwAAgoAA4GkuUmlqIzDy_PCsx4E",
    "CAACAgEAAx0CAAGgr9AAAgmWX7b6uFeLlhXEgYrM8pIbGaQKRQ0AAswBAALjeAQAAbeooNv_tb6-HgQ",
    "AgADzAEAAuN4BAAB",
    "AgACAgIAAx0CAAGgr9AAAgmZX7b7IPLRl8NcV3EJkzHwI1gwT-oAAq2nMRuBpLlJPJY-"
    "URZfhTkgfeqKEAADAQADAgADbQADAZ8BAAEeBA",
    "AQADIH3qihAAAwGfAQAB",
    "AgACAgIAAx0CAAGgr9AAAgmZX7b7IPLRl8NcV3EJkzHwI1gwT-oAAq2nMRuBpLlJPJY-"
    "URZfhTkgfeqKEAADAQADAgADeQAD_54BAAEeBA",
    "AQADAgADrKcxGylBBQAJIH3qihAAAwIAAylBBQAF7bDHYwABnc983KcAAh4E",
    "AQADAgATIH3qihAAAwIAA3t3-P______AAjhngEAAR4E",
    "AQADAgATIH3qihAAAwIAA-fFwCoX____MvARg8nvpc3RpwACHgQ",
    "BQADBAADQNKSZqjl5DcROGn_eu5JtgAEAgAEAg",
)

# One timed sample walks the corpus this many times. File ids are short, so a
# media listing is a few thousand encode calls, not one call.
COPIES = 1200
WARMUP = 3
SAMPLES = 9


def corpus() -> list[bytes]:
    raw = [rle_decode(b64_decode(file_id)) for file_id in FILE_IDS]
    for payload in raw:
        if rle_decode(rle_encode(payload)) != payload:
            raise SystemExit("rle_encode round trip changed a file-id payload")
    return raw * COPIES


def main() -> None:
    payloads = corpus()
    gc.collect()
    gc.disable()
    try:
        for _ in range(WARMUP):
            for payload in payloads:
                rle_encode(payload)

        samples: list[float] = []
        for _ in range(SAMPLES):
            started = time.perf_counter()
            consumed = 0
            for payload in payloads:
                consumed += len(rle_encode(payload))
            samples.append((time.perf_counter() - started) * 1000.0)
            if consumed <= 0:
                raise SystemExit("rle_encode produced no output")
    finally:
        gc.enable()

    print(f"median_ms={statistics.median(samples):.4f}")


if __name__ == "__main__":
    main()
