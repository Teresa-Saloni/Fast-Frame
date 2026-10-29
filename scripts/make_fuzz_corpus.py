#!/usr/bin/env python3
"""Seed the libFuzzer corpus with valid FFR1 files of a few geometries."""
import struct, zlib, pathlib
out = pathlib.Path(__file__).resolve().parent.parent / "fuzz" / "corpus"
out.mkdir(parents=True, exist_ok=True)
def crc32(b): return zlib.crc32(b) & 0xFFFFFFFF
for i, (w, h, n) in enumerate([(1, 1, 1), (2, 2, 3), (4, 3, 2), (16, 9, 1), (7, 5, 0)]):
    hdr = b"FFR1" + struct.pack("<HHIIIII", 1, 0, w, h, 30, 1, n)
    hdr += struct.pack("<I", crc32(hdr))
    body = b""
    for f in range(n):
        body += struct.pack("<IQ", w * h * 3, f * 33) + bytes([f % 256]) * (w * h * 3)
    (out / f"seed{i}.ffr").write_bytes(hdr + body)
print(f"wrote {len(list(out.iterdir()))} seeds to {out}")
