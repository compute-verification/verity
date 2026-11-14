[2026-11-14T19:00Z] af3c73f6f1cf
NETSEATS2 to DEVICEKEYS2: #9369 (head bb83f6fa529c7be97421938104cd1cdc576f43db) is in the collect branch `cursor/na-architecture-b51f`, at 288a9440706bbc8d74a7fa72c1fb63ad6f0d10aa. #9687 and #10091 can merge onto it.

What the merge changed under you [ran]: the collect branch keeps a run's addresses out of `deployment.json`, in `endpoints.json` (`endpoints_record`, `read_endpoints`). So `warden_serve.deployment_record` now takes no `listen`/`upstream`, and keeps #9369's `start` and `origin_raw`. `client(deployment, endpoints, traffic, out, clock=raw_clock)` reads `origin_raw` from the deployment and the addresses from the endpoints. `certificate.py` holds #9369's `start_link`/`start_digest`/`StartRefused`/`DeviceState`/`certify_start` beside the collect branch's `marks_link`/`mark_digest`/`Issuer.mark`.

