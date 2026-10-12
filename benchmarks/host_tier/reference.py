"""Recompute loop.cu's gate receipts from their definitions (stdlib only) and compare: `reference.py out/gate.json`.

Exit 0 if the GPU receipts, the CPU sketches and the CPU leaf all match, 3 otherwise.
"""

import hashlib
import json
import struct
import sys

M64 = (1 << 64) - 1
M32 = (1 << 32) - 1


def splitmix64(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & M64
    return x ^ (x >> 31)


def content(region, words):
    return struct.pack(f"<{words}Q", *(splitmix64(i ^ (region << 48)) for i in range(words)))


def mask(s, z):
    """The 4 KiB a receipt XORs into every block under z."""
    h = s
    for t in z:
        h = splitmix64(h ^ t)
    return struct.pack("<512Q", *(splitmix64(h ^ (0x6D00000000 + i)) for i in range(512)))


def gpu_receipts(s, region, blocks, z):
    keys = [splitmix64(s ^ (0x6B00000000 + i)) for i in range(1024)]
    k0, k1 = [(v & M32) | 1 for v in keys], [(v >> 32) | 1 for v in keys]
    words = struct.unpack(f"<{blocks * 1024}I", content(region, blocks * 512))
    m = struct.unpack("<1024I", mask(s, z))
    out = []
    for b in range(blocks):
        x = [w ^ m[i] for i, w in enumerate(words[b * 1024 : (b + 1) * 1024])]
        r0 = sum(a * k for a, k in zip(x, k0)) & M32
        r1 = sum(a * k for a, k in zip(x, k1)) & M32
        out.append(f"{r0 | r1 << 32:016x}")
    return out


def cpu_group(seed, s, region, z, zseq, group, ks):
    keys = [
        [struct.unpack("b", bytes([(splitmix64(s ^ (0x6300000000 + l * 32 + j // 8)) >> (8 * (j % 8))) & 0xFF]))[0] for j in range(256)]
        for l in range(ks)
    ]
    m = mask(s, z)
    x = content(region, (group + 1) * 8192)[group * 65536 :]
    sk = []
    for b in range(16):
        d = [v ^ m[j] for j, v in enumerate(x[b * 4096 : (b + 1) * 4096])]
        for key in keys:
            sk.append(sum(v * key[j % 256] for j, v in enumerate(d)))
    msg = seed + struct.pack("<QQ", group, zseq) + struct.pack(f"<{len(sk)}i", *sk)
    return sk, hashlib.sha256(msg).hexdigest()


def main(path):
    g = json.load(open(path))
    seed = bytes.fromhex(g["seed"])
    s = struct.unpack("<Q", seed[:8])[0]
    z0 = [splitmix64(s ^ (0x7A00000000 + t)) & M32 for t in range(16)]
    gpu = gpu_receipts(s, g["gpu_region"], len(g["gpu_receipts"]), z0)
    sk, leaf = cpu_group(seed, s, g["cpu_region"], z0, 0, 0, g["ks"])
    checks = {"gpu_receipts": gpu == g["gpu_receipts"], "cpu_sketches": sk == g["cpu_sketches"], "cpu_leaf": leaf == g["cpu_leaf"]}
    print(json.dumps({"gate": all(checks.values()), **checks}))
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
