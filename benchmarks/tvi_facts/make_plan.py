"""Write the plans: smoke.json, and a.json / b.json, one per GPU, splitting the state-size grid between them."""
import json
import sys

S_GRID = {"sub10kbit": 10_000, "sub10kB": 80_000, "sub100kB": 800_000, "sub1MB": 8_000_000}
SWEEP_LR = {"full": [2e-5, 1e-4], "lora16": [5e-4, 2e-3], "sub": [3e-4, 1e-3, 3e-3]}
SPLIT = {"a": (["full", "sub100kB"], ["sub10kbit"]), "b": (["lora16", "sub1MB"], ["sub10kB"])}


def task(group, n, lr, epochs, seed=0, bs=64, **kw):
    method = "full" if group == "full" else "lora" if group == "lora16" else "sub"
    tag = f"lr{lr:g}" if isinstance(lr, float) else "best"
    t = {"name": f"n{n}_{group}_{tag}_s{seed}", "n": n, "method": method,
         "lr": lr, "epochs": epochs, "seed": seed, "bs": bs, "group": group, **kw}
    if method == "sub":
        t["s_bits"] = S_GRID[group]
    return t


def plan(heavy, light, n_mid, n_big, n_small, e_mid, e_big, e_small, bs_big):
    """sweeps at n_mid (heavy groups first), n_big for the heavy groups, a second n_mid seed, n_small, then n_big
    for the light groups (S far below H there, so they only confirm the ceiling)"""
    groups = heavy + light
    tasks = []
    for g in heavy:
        for lr in SWEEP_LR["sub" if g.startswith("sub") else g]:
            tasks.append(task(g, n_mid, lr, e_mid, sweep=True))
    for g in heavy:
        tasks.append(task(g, n_big, f"best:{g}", e_big, bs=bs_big, analyze=True, save=True))
    for g in light:
        for lr in SWEEP_LR["sub"]:
            tasks.append(task(g, n_mid, lr, e_mid, sweep=True))
    for g in groups:
        tasks.append(task(g, n_mid, f"best:{g}", e_mid, seed=1, analyze=True, save=True))
    for g in groups:
        for seed in (0, 1):
            tasks.append(task(g, n_small, f"best:{g}", e_small, seed=seed, bs=16, analyze=seed == 0))
    for g in light:
        tasks.append(task(g, n_big, f"best:{g}", e_big, bs=bs_big, analyze=True))
    return {"slots": 3, "tasks": tasks}


if __name__ == "__main__":
    n_big = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    e_big = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    for k, (heavy, light) in SPLIT.items():
        json.dump(plan(heavy, light, 2000, n_big, 200, 20, e_big, 60, 128), open(f"{k}.json", "w"), indent=1)
    smoke = {"slots": 2, "tasks": [
        task("full", 200, 1e-4, 1, analyze=True, save=True),
        task("sub1MB", 200, 1e-3, 1, analyze=True, save=True),
    ]}
    json.dump(smoke, open("smoke.json", "w"), indent=1)
