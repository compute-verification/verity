[2026-11-12T14:28Z] 3bb6767d4422
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 6:28 AM PST. **vllm has answered the question that held NET-368 open on its side: its readers take the device's own buckets, and the device hands out nothing new for them (NET-340, 6:21 AM PST).** Its lead's own line follows by Fri 13 Nov noon PST.

- So nothing is added to NET-368 for vllm. Keep `origin_unix` in `deployment.json` exactly as you have it: vllm's load starts by it and its chain reads it for which Commit windows are in a run.
- The origin on the device's clock stays too; it is for our own rehearsal's client. vllm's readers do not use it.
- The mark still waits for lean's merge on `main`, and for tonight's serve to have run on the device as `main` has it.

