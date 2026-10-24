"""Write the plans: smoke.json, and a.json / b.json / c.json, one per GPU, splitting the state-size grid.

Every N trains for the same number of epochs (E_MID = E_BIG), so a fact is seen equally often whatever N is;
N_SMALL trains longer because its epochs are only a few steps.
"""
import json
import sys

S_GRID = {"sub10kbit": 10_000, "sub10kB": 80_000, "sub100kB": 800_000, "sub1MB": 8_000_000}
SWEEP_LR = {"full": [2e-5, 1e-4], "lora16": [5e-4, 2e-3], "sub": [3e-4, 1e-3, 3e-3]}
N_SMALL, N_MID, N_BIG = 200, 2000, 20000
E_SMALL, E_MID, E_BIG = 30, 12, 12
# per GPU: groups swept at N_MID and run at N_SMALL and N_MID (seed 1), then the groups run at N_BIG
SPLIT = {"a": (["full", "sub10kbit"], ["full", "sub10kbit"]),
         "b": (["lora16", "sub10kB"], ["lora16", "sub10kB"]),
         "c": (["sub1MB", "sub100kB"], ["sub1MB", "sub100kB"]),
         "d": (["tlora1MB", "tlora100kB", "tlora10kB"], ["tlora1MB", "tlora100kB"])}
BIG_LAST = {"sub10kB", "sub10kbit"}
ALL_MAPS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MLP = ["gate_proj", "up_proj", "down_proj"]
# tiny LoRA sized to S: rank 1 on chosen maps, trained directly (S = 16 bits x parameters)
TLORA = {"tlora10kB": {"layers": [12], "maps": ["down_proj"], "rank": 1},              # 5,760 params, 11.5 kB
         "tlora100kB": {"layers": [8, 12, 16], "maps": MLP, "rank": 1},                # 51,840 params, 104 kB
         "tlora1MB": {"layers": list(range(24)), "maps": ALL_MAPS, "rank": 1}}         # 549,888 params, 1.1 MB
SWEEP_LR["tlora"] = [5e-4, 1.5e-3]  # S far below H at N_BIG: they only confirm the ceiling


def task(group, n, lr, epochs, seed=0, bs=64, **kw):
    method = ("full" if group == "full" else "lora" if group == "lora16" else "tlora" if group.startswith("tlora")
              else "sub")
    tag = f"lr{lr:g}" if isinstance(lr, float) else "best"
    t = {"name": f"n{n}_{group}_{tag}_s{seed}", "n": n, "method": method,
         "lr": lr, "epochs": epochs, "seed": seed, "bs": bs, "group": group, **kw}
    if method == "sub":
        t["s_bits"] = S_GRID[group]
    if method == "tlora":
        t["tlora"] = TLORA[group]
    return t


def plan(mid, big):
    first = [g for g in big if g not in BIG_LAST]
    tasks = []
    for g in sorted(mid, key=lambda g: g not in first):
        for lr in SWEEP_LR["sub" if g.startswith("sub") else "tlora" if g.startswith("tlora") else g]:
            tasks.append(task(g, N_MID, lr, E_MID, sweep=True))
    for g in first:
        tasks.append(task(g, N_BIG, f"best:{g}", E_BIG, bs=128, analyze=True, save=True))
    for g in mid:
        tasks.append(task(g, N_MID, f"best:{g}", E_MID, seed=1, analyze=True, save=True))
    for g in mid:
        for seed in (0, 1):
            tasks.append(task(g, N_SMALL, f"best:{g}", E_SMALL, seed=seed, bs=16, analyze=seed == 0))
    for g in big:
        if g in BIG_LAST:
            tasks.append(task(g, N_BIG, f"best:{g}", E_BIG, bs=128, analyze=True))
    return {"slots": 3, "tasks": tasks}


if __name__ == "__main__":
    for k, (mid, big) in SPLIT.items():
        json.dump(plan(mid, big), open(f"{k}.json", "w"), indent=1)
    if len(sys.argv) > 1:
        # a second model at N_MID: `make_plan.py PREFIX GROUP=LR ...` -> PREFIX.json, one seed per group
        prefix, tasks = sys.argv[1], []
        for gl in sys.argv[2:]:
            g, lr = gl.split("=")
            t = task(g, N_MID, float(lr), E_MID, analyze=True, save=True)
            t["name"] = f"{prefix}_{t['name']}"
            tasks.append(t)
        json.dump({"slots": 3, "tasks": tasks}, open(f"{prefix}.json", "w"), indent=1)
    smoke = {"slots": 2, "tasks": [
        task("full", 200, 1e-4, 1, analyze=True, save=True),
        task("sub1MB", 200, 1e-3, 1, analyze=True, save=True),
    ]}
    json.dump(smoke, open("smoke.json", "w"), indent=1)
