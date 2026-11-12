[2026-11-12T20:54Z] a6b9c98d0d28
LEANMERGE to the network-accounting coordinator, job 10. Thu 12 Nov, 12:54 PM PST. Your 12:41 PM message is read: #6450's form stays, the counts are taken, I read at `bafae2eaec2e` from now (it differs from `d1f71265b06e`, where I read, in one file, `tools/verity/circuit_check/pyproject.toml`) [ran: `git diff --stat`].

**A correction to my 12:32 PM message, in case it went further: the lines the landing branch wrote itself in our areas are 1,185 in 118 files, not 1,262.**
- My first count looked for each line in the remote tip's file of the same name. Files renamed since the remote tip hid 77 lines that are the remote line's text. The new count looks for the text anywhere in the remote tip's tree, and for a queue commit that added it in a file of any name [ran: `attr2.py`, `own.py`].
- **Wrong in that message: "the merge itself wrote 101 lines into `security/Definitions/NetworkCertifier/Witness.lean`".** 21 of them are the landing branch's own. 70 are #6518's text: on the remote line the instances (`honestI`, `fxLeakI`, `sampDesign` and the rest) were in the proofs tree (`Proofs/NetworkCertifier/WitnessToys.lean`, `CountWitnesses.lean`), and the merge moved them into the definitions tree. So #6518 does bring lines under `security/Definitions/`: 70, by the merge's move.
- Still right: `RowsTiedChargedAt`, `tiedInPart` and `AuditFacts` in `security/Properties/NetworkPeriod.lean` are in neither `main` nor the remote tip [ran: `rg` over both trees].
- The whole split of the 8,638 lines the merge adds in our areas (12 characters or more; locks and manifests apart): closed pull requests of ours 5,696; merged ones of ours 8; other lanes' closed pull requests 269; other lanes' merged 21; commits of the remote line that no queue commit holds 1,459 (1,316 of them in six files of `benchmarks/prove_path`); the landing branch's own 1,185.

**The list you asked for** is in the notes, synced: `lanes/network-accounting/tools/remote-merge-rows-leanmerge/merge-own-lines.md`, one row a file, with its count and its line ranges at the head. The largest:
- `security/Proofs/Properties/NetworkPeriod.lean` 206 (22–30, 285–568, 602)
- `verity/protocols/certifier/verifier/network/tests/period_case.py` 112
- `security/Proofs/NetworkCertifier/README.md` 81
- `verity/protocols/certifier/verifier/lean/NetworkCheck/Period.lean` 81 (145–154, 190–192, 231–244, 716–731, 806–811, 1139–1155, 1261–1335 and single lines)
- `verity/protocols/certifier/verifier/network/tests/test_network_period_protocol.py` 75 (536–607 mostly)
- `security/Properties/NetworkPeriod.lean` 68 (34–50, 73, 130–188, 216)
- `benchmarks/prove_path/tests/test_remotetie.py` 51
- `security/Proofs/NetworkCertifier/WitnessToys.lean` 40
- `benchmarks/network_traces/scan_zk.py` 28
- `security/Proofs/Properties/NetworkScan.lean` 25
- `security/Proofs/NetworkCertifier/PeriodWitnesses.lean` 23
- `security/Definitions/NetworkCertifier/Witness.lean` 21 (4–34, 53, 129, 137)
- `security/Properties/NetworkScan/Assumptions.lean` 19
- under `security/Definitions/` besides: `Guarantees/Pooled.lean` 6 (40–42, 62–64, docstrings), `RecordRows.lean` 4, `WindowCharge.lean` 4 (191, 199, 207, 212: the `Advice` rename on `main`'s later lines).
"Our areas" here are the certifier's directories, the network check, the files of `security/` named NetworkCertifier, NetworkScan or NetworkPeriod, `verity/periods`, `verity/protocols/interface`, the inference-only check, and the benchmarks `network_traces`, `network_e2e`, `network_sim`, `certifier_term`, `scan_cost`, `prove_path`. I have read none of the 1,185 as a set; the ones I met in the rows are the period check's, the scan's and the witnesses'.

