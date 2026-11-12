"""ccmem_order.py DIR: compare the registered roots' order of circuit-check's own process (DIR/cli.json) with a session's
workers' (DIR/session/gw*.json) and with those of a session that loaded the registries first (DIR/first/gw*.json)."""
import json
import pathlib
import sys

RECORDS = 8  # units.REGISTERED_RECORDS


def chunks(ids: list) -> list:
    k = max(1, -(-len(ids) // RECORDS))
    return [ids[j:j + k] for j in range(0, max(1, len(ids)), k)]


d = pathlib.Path(sys.argv[1])
try:
    cli = json.loads((d / "cli.json").read_text())
except (OSError, ValueError) as e:
    print(f"   ORDER circuit-check's own process: not read ({e})")
    sys.exit(0)
print(f"   ORDER circuit-check's own process: {len(cli)} registered roots, {len(chunks(cli))} records of {len(chunks(cli)[0])}")
for name in ("session", "first"):
    files = sorted((d / name).glob("gw*.json"))
    if not files:
        print(f"   ORDER {name}: no worker wrote its order")
    for p in files:
        got = json.loads(p.read_text())
        first = next((i for i, (a, b) in enumerate(zip(cli, got)) if a != b), None)
        differ = [i for i, (a, b) in enumerate(zip(chunks(cli), chunks(got))) if a != b]
        print("   ORDER", json.dumps({
            "where": f"{name}/{p.stem}", "roots": len(got), "same_order": got == cli, "same_set": set(got) == set(cli),
            "only_here": sorted(set(got) - set(cli))[:12], "only_in_circuit_checks": sorted(set(cli) - set(got))[:12],
            "first_position_that_differs": first, "there": None if first is None else [cli[first], got[first]],
            "records_whose_roots_differ": differ}))
