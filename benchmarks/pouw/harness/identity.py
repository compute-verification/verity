"""Which GPU a run is on, checked before anything else: `CUDA_VISIBLE_DEVICES` names exactly one index, the worker's; CUDA
sees exactly one device; its UUID is the one expected for that index (`--expect-uuid`); and CUDA orders devices by PCI
bus id, so CUDA and NVML agree. NVML is addressed by that UUID, never by an index."""

from __future__ import annotations

import re
from pathlib import Path

CARD_FILES = Path(__file__).resolve().parents[3] / "calibration/hardware/cards"

#: what `bench.py`, `characterize.py`, `native.py` and the tests read
__all__ = ["IdentityError", "check", "check_card", "expected_for", "parse_server_md", "resolve_gpu", "uuid_string"]

UUID_RE = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


class IdentityError(RuntimeError):
    pass


def uuid_string(raw: bytes) -> str:
    """CUDA's 16 UUID bytes in NVML's form."""
    h = raw.hex()
    return f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def parse_server_md(text: str) -> dict[int, str]:
    """The GPU index → UUID table of the server page: every table row whose first cell is an index and that holds a UUID."""
    out: dict[int, str] = {}
    for line in text.splitlines():
        cells = [c.strip().strip("`") for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not cells[0].isdigit():
            continue
        found = UUID_RE.search(line)
        if found:
            out[int(cells[0])] = found.group(0)
    return out


def resolve_gpu(arg: str, env: dict[str, str]) -> int:
    """`--gpu`: an index, or `lease`, the one index a lease put in `CUDA_VISIBLE_DEVICES` (the RTX PRO 6000 node's
    `gpu-lease 1` takes the first free GPU, so the index is known only inside the lease)."""
    if arg != "lease":
        return int(arg)
    cvd = (env.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not cvd.isdigit():
        raise IdentityError(f"--gpu lease needs CUDA_VISIBLE_DEVICES to be one index, is {cvd!r}")
    return int(cvd)


def expected_for(gpu_arg: str, expect_uuid: str | None, server_md: str | None, env: dict[str, str]) -> tuple[int, str | None]:
    """The run's GPU index and the UUID(s) it must have: `--expect-uuid`, else that index's row of the server table."""
    gpu = resolve_gpu(gpu_arg, env)
    if expect_uuid or server_md is None:
        return gpu, expect_uuid
    table = parse_server_md(server_md)
    if gpu not in table:
        raise IdentityError(f"the server table has no GPU {gpu}")
    return gpu, table[gpu]


def check(env: dict[str, str], gpu: int, device_count: int, uuid: str, expected_uuid: str | None,
          pod: bool = False) -> dict:
    """Refuses the run unless it is on GPU `gpu` alone and that GPU is `expected_uuid`. `expected_uuid` may list several,
    comma-separated: a queue job that can't pin an index (the RTX PRO 6000 node's Kueue queue gives a one-GPU job
    whichever GPU is free, as index 0) passes the node's UUIDs and records which one it got. On a single-GPU pod
    (`pod`), which has no server table, the UUID is recorded and matched only when one is given."""
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd.strip() != str(gpu):
        raise IdentityError(f"CUDA_VISIBLE_DEVICES must be exactly {gpu!r}, is {cvd!r}")
    if env.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        raise IdentityError("CUDA_DEVICE_ORDER must be PCI_BUS_ID")
    if device_count != 1:
        raise IdentityError(f"CUDA sees {device_count} devices, not 1")
    if not expected_uuid and not pod:
        raise IdentityError("no expected UUID (--expect-uuid, from server.md's table)")
    expected = [u.strip() for u in (expected_uuid or "").split(",") if u.strip()]
    for u in expected:
        if not UUID_RE.fullmatch(u):
            raise IdentityError(f"malformed expected UUID {u!r}")
    if expected and uuid not in expected:
        want = f"one of {', '.join(expected)}" if len(expected) > 1 else f"GPU {gpu}'s {expected[0]}"
        raise IdentityError(f"the visible device is {uuid}, not {want}")
    return {"gpu": gpu, "uuid": uuid, "uuid_checked": bool(expected), "expected_uuids": expected, "pod": pod,
            "cuda_visible_devices": cvd, "cuda_device_order": env["CUDA_DEVICE_ORDER"]}


def _release(driver: str) -> tuple[int, ...]:
    return tuple(int(p) if p.isdecimal() else -1 for p in driver.split("."))


def check_card(name: str, capability: str, sms: int, driver: str) -> dict:
    """Refuses the run unless the device is the card file (calibration/hardware/cards/) whose `driver` holds `name`, the
    name the device gives: its `arch`'s architecture file's (calibration/hardware/architectures/) compute capability, its
    `sm.count` SMs, and a driver no older than the oldest `driver_min` of that architecture's toolkit gates. The build
    target is the card's `arch`."""
    # here: loading the harness needs the standard library alone
    from verity.device import gpu_card_reporting, load_architecture, number, skus

    try:
        card = gpu_card_reporting(name, directory=CARD_FILES)
        architecture = load_architecture(card["arch"], directory=CARD_FILES.parent / "architectures")
    except KeyError as e:
        raise IdentityError(f"not the card: {e.args[0]} (the GPU card files: "
                            f"{', '.join(skus('gpu', directory=CARD_FILES))})") from None
    arch, want_capability = card["arch"], architecture["capability"]
    min_driver = min((gate["driver_min"] for gate in architecture["toolkit"]), key=_release)
    want_sms = int(number(card["sm"]["count"]))
    problems = []
    if capability != want_capability:
        problems.append(f"compute capability {capability}, not {want_capability}")
    if sms != want_sms:
        problems.append(f"{sms} SMs, not {want_sms}")
    if _release(driver) < _release(min_driver):
        problems.append(f"driver {driver!r}, need {min_driver} or later")
    if problems:
        raise IdentityError("not the card: " + "; ".join(problems))
    return {"name": name, "capability": capability, "sms": sms, "driver": driver, "arch": arch}
