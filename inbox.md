[2026-11-14T19:48Z] f614c8ff2454
## Coordinator, Sat 14 Nov 19:48Z: CONNECTED: 85-rec-reprice.sh

From the proofs lead, a third thing from your rename (benchmarks, TOP-424 19:49Z): in the pod script `85-rec-reprice.sh`, line 1028 creates `<statement>-zk_server` and line 1104 reads `<statement>-zk-server/prove.err`, so with `RELEASE=1` the release report finds no `ZK-SERVER-REFUSED`. One spelling was `firewall` on both lines before. Make the two agree (the directory name the script creates is the one to keep unless another reader needs the other), then `rg` your whole rename for the same split: every place a path or key is WRITTEN with `zk_server` and READ with `zk-server` or the reverse. `bash -n` the script; commit in your next note's first lines. Order: `parties.py:519` first, this second, the map test third, then the builds.

