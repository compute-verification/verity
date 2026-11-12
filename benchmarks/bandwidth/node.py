"""What a node reports of itself, for its node file (`verity/hardware/nodes/`): the interfaces that carry traffic off
it, with their line rates, and what each NVIDIA card's firmware says it gives a program. One Attempt on the node (cwd =
the source tree), which writes nothing to a device:

    research run --on NODE --project verity --source . --cwd source -- \
      bash -c 'python3 benchmarks/bandwidth/node.py --out "$OUT"'

**Interfaces** (`net.json`, and the commands' own text in `net.txt`). For every interface under `/sys/class/net`: its
`operstate`, `speed` (Mb/s; a virtual interface often reads -1 or refuses the read), `type` and driver, and `ethtool
IF`; beside them `ip -d -s link`, `ip route` and `ip -6 route`, and each InfiniBand port's rate. An interface leaves
the node (`leaves`) when it is a device's (`/sys/class/net/IF/device` exists: the loopback, a bridge, a veth pair and a
tunnel are the node's own software and have none) and a route of the main table, of either family, names it: such a
route sends packets out of the node through it, and an interface no route names carries none. `net.json` holds no
address.

**Registers** (`regs.json`, with `nvidia-smi -q` in `nvidia-smi.txt`). Over each NVIDIA card's BAR0, mapped read-only
(`/sys/bus/pci/devices/BDF/resource0`, which only root may map: without it the script asks `sudo -n` and, refused,
records the refusal), at the addresses NVIDIA's open kernel modules give at the tag `TAG`:

- `NV_PMC_BOOT_0` and `NV_PMC_BOOT_42`, which name the chip (`nv_ref.h`);
- `NV_USABLE_FB_SIZE_IN_MB`, the memory the vBIOS's devinit publishes as usable, in MiB
  (`NV_PGC6_AON_SECURE_SCRATCH_GROUP_42`, `ampere/ga102/dev_gc6_island.h` and its addendum; the driver reads it there
  for every chip from GA102 on, Hopper's and Blackwell's among them: `kmemsysReadUsableFbSize_GA102`);
- `NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE`, the MMU's local memory range, magnitude `<<` scale in MiB, on the chips whose
  header defines it at the tag: Blackwell's GB1xx (`blackwell/gb100/dev_fb.h`; `kern_mem_sys_gb100.c` decodes it).
  No header at the tag defines it for GH100 or for GB2xx, so on those it is read only at each address the caller
  gives, and the output says the address is the caller's and decodes the value both ways the tag has. The two
  addresses the tag has for it are `--range-at 0x001FA3E0`, GB100's, in the block GH100's own
  `NV_PFB_PRI_MMU_WPR1_ADDR_LO`, 0x001FA81C, sits in, and `--range-at 0x00100CE0`, Pascal's
  (`pascal/gp102/dev_fb.h`), which the driver reads up to GA100 (`kmemsysReadUsableFbSize_GP102`).

`--regs must` exits 3 unless every card was read; the default records a refusal and goes on. Outputs under `--out`,
the run's `$OUT`: `net.txt`, `net.json`, `regs.json`, `nvidia-smi.txt`.
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

TAG = "580.173.02"
"""The tag of NVIDIA's open kernel modules (github.com/NVIDIA/open-gpu-kernel-modules) the addresses below are read
from, under `src/common/inc/swref/published/`."""

NV_PMC_BOOT_0 = 0x00000000
NV_PMC_BOOT_42 = 0x00000A00
NV_USABLE_FB_SIZE_IN_MB = 0x001183A4
NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE_GB100 = 0x001FA3E0

ARCHITECTURES = {0x17: "Ampere", 0x18: "Hopper", 0x19: "Ada", 0x1A: "Blackwell GB1xx", 0x1B: "Blackwell GB2xx"}
"""`NV_PMC_BOOT_42_ARCHITECTURE` (bits 29:24) of the chips whose usable-memory register is the one above (`nv_arch.h`;
of Ampere, every chip but GA100, implementation 0, which reads the range register at Pascal's address)."""

MAP = 0x200000


def ranges(value: int) -> dict:
    """A local memory range register's value under each layout the tag has: GB100's (magnitude in bits 27:4) and
    GP102's (magnitude in bits 9:4, and fifteen sixteenths of it with ECC, bit 30), each `<<` the scale of bits 3:0."""
    scale = value & 0xF
    gp102 = ((value >> 4) & 0x3F) << scale
    return {"raw": f"{value:#010x}", "scale": scale, "MiB_gb100": ((value >> 4) & 0xFFFFFF) << scale,
            "MiB_gp102": gp102 * 15 // 16 if value >> 30 & 1 else gp102}


def run(*cmd: str) -> str:
    """A command's output, or the line that says why there is none."""
    if shutil.which(cmd[0]) is None:
        return f"{cmd[0]}: not installed\n"
    done = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return done.stdout + done.stderr + (f"[exit {done.returncode}]\n" if done.returncode else "")


def read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError as e:
        return f"refused: {e.strerror}"


def interfaces() -> tuple[dict, str]:
    """Each interface's own report and whether it leaves the node, with the text of every command read."""
    if shutil.which("ip") is None:
        raise SystemExit("REFUSED: no `ip` on this node, so no route says which interface leaves it (install iproute2)")
    text = []
    routes = {}
    for family in ("-4", "-6"):
        out = run("ip", family, "route", "show", "table", "main")
        text.append(f"$ ip {family} route show table main\n{out}")
        routes[family] = [line for line in out.splitlines() if re.search(r"\bdev (\S+)", line)]
    text.append("$ ip -d -s link\n" + run("ip", "-d", "-s", "link"))
    rows = {}
    for path in sorted(Path("/sys/class/net").iterdir()):
        name = path.name
        named = {f: [r for r in routes[f] if re.search(rf"\bdev {re.escape(name)}\b", r)] for f in routes}
        driver = path / "device" / "driver"
        device = (path / "device").exists()
        ethtool = run("ethtool", name)
        text.append(f"$ ethtool {name}\n{ethtool}")
        speed = re.search(r"^\s*Speed: (.+)$", ethtool, re.M)
        row = {
            "operstate": read(path / "operstate"),
            "speed": read(path / "speed"),
            "type": read(path / "type"),
            "driver": os.path.basename(os.readlink(driver)) if driver.is_symlink() else None,
            "device": device,
            "ethtool_speed": speed.group(1).strip() if speed else None,
            "routes": {f: len(named[f]) for f in named},
            "default": {f: any(r.startswith("default") for r in named[f]) for f in named},
        }
        row["leaves"] = device and any(named.values())
        rows[name] = row
    ports = {}
    for port in sorted(Path("/sys/class/infiniband").glob("*/ports/*")):
        ports[f"{port.parent.parent.name}/{port.name}"] = {k: read(port / k) for k in ("rate", "state", "link_layer")}
    return {"interfaces": rows, "infiniband": ports}, "\n".join(text)


def cards() -> list[Path]:
    """The NVIDIA display and 3D controllers on the PCI bus."""
    out = []
    for path in sorted(Path("/sys/bus/pci/devices").iterdir()):
        if read(path / "vendor") == "0x10de" and (read(path / "class") or "")[:4] in ("0x03",):
            out.append(path)
    return out


def card(path: Path, range_at: list[int]) -> dict:
    """One card's registers over its BAR0, mapped read-only."""
    row: dict = {"bdf": path.name, "device": read(path / "device")}
    start, end, _ = ((path / "resource").read_text().splitlines()[0]).split()
    if int(end, 16) - int(start, 16) + 1 < MAP:
        return row | {"refused": "BAR0 is under 2 MiB"}
    fd = os.open(path / "resource0", os.O_RDONLY)
    try:
        bar = mmap.mmap(fd, MAP, mmap.MAP_SHARED, mmap.PROT_READ)
    finally:
        os.close(fd)
    words = memoryview(bar).cast("I")
    try:
        boot42 = words[NV_PMC_BOOT_42 // 4]
        arch, impl = (boot42 >> 24) & 0x3F, (boot42 >> 20) & 0xF
        row |= {
            "NV_PMC_BOOT_0": f"{words[NV_PMC_BOOT_0 // 4]:#010x}",
            "NV_PMC_BOOT_42": f"{boot42:#010x}",
            "architecture": ARCHITECTURES.get(arch, f"{arch:#04x}"),
            "implementation": impl,
        }
        if arch in ARCHITECTURES and (arch, impl) != (0x17, 0):
            usable = words[NV_USABLE_FB_SIZE_IN_MB // 4]
            row["NV_USABLE_FB_SIZE_IN_MB"] = {"at": f"{NV_USABLE_FB_SIZE_IN_MB:#010x}", "raw": f"{usable:#010x}",
                                              "MiB": usable}
        else:
            row["NV_USABLE_FB_SIZE_IN_MB"] = {"refused": "the driver reads no usable size there for this chip"}
        if arch == 0x1A:
            row["NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE"] = [
                {"at": f"{NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE_GB100:#010x}", "address": f"blackwell/gb100/dev_fb.h at {TAG}"}
                | ranges(words[NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE_GB100 // 4])]
        elif range_at:
            row["NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE"] = [
                {"at": f"{at:#010x}", "address": "the caller's: no header at the tag defines it for this chip"}
                | ranges(words[at // 4]) for at in range_at]
        else:
            row["NV_PFB_PRI_MMU_LOCAL_MEMORY_RANGE"] = {
                "refused": "no header at the tag defines it for this chip, and no --range-at was given"}
    finally:
        words.release()
        bar.close()
    return row


def registers(range_at: list[int]) -> dict:
    """Every card's registers, read here as root or through `sudo -n`, or the refusal."""
    found = cards()
    out: dict = {"tag": TAG, "driver": read(Path("/sys/module/nvidia/version")), "cards": []}
    if not found:
        return out
    if os.geteuid() != 0:
        again = ["sudo", "-n", sys.executable, os.path.abspath(__file__), "--regs-json"]
        again += [arg for at in range_at for arg in ("--range-at", hex(at))]
        done = subprocess.run(again, capture_output=True, text=True, check=False)
        if done.returncode:
            return out | {"refused": f"not root, and sudo -n: {(done.stderr or done.stdout).strip()[:200]}"}
        return json.loads(done.stdout)
    for path in found:
        try:
            out["cards"].append(card(path, range_at))
        except OSError as e:
            out["cards"].append({"bdf": path.name, "refused": e.strerror})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", help="the directory the outputs go in; not needed with --regs-json")
    ap.add_argument("--regs", choices=("try", "must", "no"), default="try")
    ap.add_argument("--range-at", type=lambda s: int(s, 16), action="append", default=[],
                    help="read the MMU's local memory range at this address on a chip whose header defines none")
    ap.add_argument("--regs-json", action="store_true", help="print the registers as JSON and nothing else")
    args = ap.parse_args()
    if any(at % 4 or not 0 <= at < MAP for at in args.range_at):
        ap.error("--range-at is a 32-bit register's address in BAR0's first 2 MiB")
    if args.regs_json:
        print(json.dumps(registers(args.range_at)))
        return 0
    if not args.out:
        ap.error("--out DIR: where the outputs go (a run's \"$OUT\")")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    net, text = interfaces()
    (out / "net.txt").write_text(text, encoding="utf-8")
    (out / "net.json").write_text(json.dumps(net, indent=1) + "\n", encoding="utf-8")
    leaving = {name: row for name, row in net["interfaces"].items() if row["leaves"]}
    print(json.dumps({"leaves": {n: {k: r[k] for k in ("operstate", "speed", "ethtool_speed", "driver")}
                                 for n, r in leaving.items()}, "infiniband": net["infiniband"]}))
    if args.regs == "no":
        return 0
    (out / "nvidia-smi.txt").write_text(run("nvidia-smi", "-q"), encoding="utf-8")
    regs = registers(args.range_at)
    (out / "regs.json").write_text(json.dumps(regs, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(regs))
    read_all = bool(regs["cards"]) and not any("refused" in c for c in regs["cards"])
    return 0 if read_all or args.regs == "try" else 3


if __name__ == "__main__":
    sys.exit(main())
