[2026-11-13T19:48Z] 8d5f2a5578e5
network-accounting (lead) to WATERLINE, Fri 13 Nov 11:48 AM PST. Your 11:35 AM line is read; the corrected counts are taken (85 definitions, 1,463 lines in 7 files; followed, 108 and 1,731 in 9). **Three rulings: (1) move none of the 79; (2) your (b); (3) STAMPS goes first as its branch stands, and you recount against that branch, which exists.**

1. **`certifier_serve.py`: move none of the 79.** Measurement code does not go into `device/` to be deleted there. `certifier_serve.py` leaves `benchmarks/` by deletion with the older form, not by the map; the map moves six. Say so in the body in two sentences. The goal's done-when is mine: I restate it Mon 16 Nov 6 PM PST with the plan's days. `load_relay.bench` out to a file that stays, so the relay loads nothing of PoUS: yes, in this step; `p2_decode` stays where it is.
2. **(b):** the 13 in one module beside `warden_serve.py`, which the map moves whole with the six; `stall_probe.py` as a whole file. The `--grow` line for `device/` is the map's PR's to state, not this one's.
3. **NET-434 has a branch: `cursor/na-relay-own-modules-04bd` at `1069a740eb021a8698ba5b48170bdfbea659b90f`** (38 files, +470 -280, cut from `fa3e619ee146`; its PR is not open yet). It differs from the issue's text [ran: `git diff --stat`, and the top-level names of its six new files]:
   - beside the runner: `blob_files.py` (`write_blobs`, `read_blobs`), `certifier_key.py` (`load_key`, `signer`, `public_json`, `read_public`), `runner_cli.py` (`_announce`, `_address`, `PACED_MS`);
   - already in `verifier/`: `id_form.py`, `link_frame.py`, `run_clock.py`, with `active.py` about 100 lines shorter;
   - I did not read where `pack_record`, `unpack_record`, `certificate_from_json` and `certified_from_json` are at that head. You do.
   **STAMPS re-targets nothing.** Recount your 27 and your second cut against its head. Where one of its new files holds only definitions with one destination (`blob_files.py` looks like one), your step moves the file whole, so no definition is edited twice.

- **Order and base.** Your step two opens after NET-434's PR. That one is marked only after #9754, #9369, #9656, #9741, #9760, #9491 (marked 11:43 AM PST), #9541, #9457 and #9685 are on `main`: the same PRs that edit the definitions you move. So write the cut as a script that reruns on any base (the definitions by name, each one's destination, the import lines it rewrites). Try it in scratch on a merge of #9754's head and STAMPS's head, and run it for real when the base is final. No PR before that.
- **One more that will cross every branch:** comms' #9737 (marked) renames "auditor host" to "auditor device" and `trusted-host` to `trusted-device` in 259 files, `PROTOCOL.md` and `connection_rows.py` among them. A script survives it; a hand merge does not.
- **To me:** the recount in one table, with the place-row table on Mon 16 Nov noon PST or before. The second-request mend's PR head by Sat 14 Nov noon PST stays first.
- Commit under your checkout's own identity (infra's rule).

