"""Peak memory of one pure-Python AES-256-IGE decrypt of a 1 MiB file chunk.

`Client.get_file` asks for `1024 * 1024` bytes per `upload.GetFile`. With no
kuricrypt/tgcrypto installed, `ige256_decrypt` runs `ige`, which keeps every
16-byte block in a list until the final join. The traced section is one
decrypt. Input ciphertext is allocated before tracing starts.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import random
import statistics
import sys
import tracemalloc
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

from pyrogram.crypto import aes

# Same ceiling `Client.get_file` passes as `upload.GetFile.limit`.
CHUNK = 1024 * 1024
SAMPLES = 7


def _inputs() -> tuple[bytes, bytes, bytes, bytes]:
    rng = random.Random(0)
    key = rng.randbytes(32)
    iv = rng.randbytes(32)
    plain = rng.randbytes(CHUNK)
    cipher = aes.ige256_encrypt(plain, key, iv)
    if aes.ige256_decrypt(cipher, key, iv) != plain:
        raise SystemExit("ige256_decrypt did not round-trip the file chunk")
    return key, iv, plain, cipher


def _peak(key: bytes, iv: bytes, cipher: bytes) -> int:
    gc.collect()
    tracemalloc.start()
    tracemalloc.clear_traces()
    try:
        out = aes.ige256_decrypt(cipher, key, iv)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    if len(out) != CHUNK:
        raise SystemExit("ige256_decrypt changed the chunk length")
    return peak


def main() -> None:
    if aes._crypto_backend is not None:
        raise SystemExit("expected the pure-Python AES fallback, found a native backend")

    key, iv, _plain, cipher = _inputs()
    samples = [_peak(key, iv, cipher) for _ in range(SAMPLES)]
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    print(f"median_bytes={median:.0f}")
    print(f"noise_pct={noise:.4f}")
    print("samples=" + ",".join(str(sample) for sample in samples))


if __name__ == "__main__":
    main()
