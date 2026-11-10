"""Model: the mined quiet sweep's server times convolved with a measured network leg.

    python model_mined.py SWEEP_DIR PROBE.bin=LABEL... > mined-model.json

SWEEP_DIR holds the mined sweep's `sweep-*.npz`, each with a `server_ns` array; each PROBE.bin is a remote cell whose
host-memory answers (op 2) give the network leg."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import LOST, REC, TAUS_US, convolve_late  # noqa: E402

server = np.concatenate([np.load(f)["server_ns"] for f in sorted(glob.glob(os.path.join(sys.argv[1], "sweep-*.npz")))])
out = {"n_server": int(len(server)),
       "server_p": {str(t): float((server > t * 1000).mean()) for t in TAUS_US},
       "server_late": {str(t): int((server > t * 1000).sum()) for t in TAUS_US},
       "server_max_us": float(server.max() / 1000), "model": {}}
for spec in sys.argv[2:]:
    path, label = spec.split("=")
    r = np.fromfile(path, dtype=REC)
    h = r[r["op"] == 2]
    ok = h["elapsed"] != LOST
    net = h["elapsed"][ok].astype(np.int64) - h["server"][ok].astype(np.int64)
    out["model"][label] = {"n_net": int(len(net)), "net_lost": int((~ok).sum()),
                           "p": convolve_late(server, net, int((~ok).sum()))}
print(json.dumps(out, indent=1))
