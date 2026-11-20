"""The premises each application's placement leaves unmet on one host, as `verity run` records them in `simulated`."""
import json

from verity import config as K
from verity_cli import run as R
reg = R.registry()
import served_table as T
ran = R.head()
out = {}
for a in ["remote", "onsite_network", "onsite_network_front", "onsite_compute", "onsite_full"]:
    cfg = R.load(R.APPLICATIONS / f"{a}.toml", reg)
    machines = list(dict.fromkeys(R.placed(cfg).values()))
    program = {**cfg, "program": {**cfg["program"], "commit": ran}}
    cid = K.composite(cfg["protocols"], reg.components)
    out[a] = {"protocols": cfg["protocols"], "machines": machines, "configured_program": cfg["program"]["commit"],
              "simulated_if_all_machines_one_host": R.unmet(program, reg, ran, machines),
              "sharing": R.sharing(cfg, reg, machines), "coins_on": R.coins_on(cfg, reg, machines),
              "coins": T.coins_of(cfg), "composite_id": cid, "unaccepted": cid in K.UNACCEPTED,
              "pins": [(p["party"], p["protocol"]) for p in cfg.get("pin", [])],
              "service": cfg.get("service")}
    out[a]["gate_coins_on"] = T.gate_coins_on(cfg, {})
    out[a]["trusted_device_dev"] = T.coins_on(cfg, {})
print(json.dumps(out, indent=1, sort_keys=True))
