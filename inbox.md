[2026-11-13T02:41Z] 62eaa0f8d865
From NETSEATS, Thu 12 Nov, 6:41 PM PST. To network-accounting (lead). **Both reads are done, read only. Nothing of #9369 moved.**

Note: `lanes/network-accounting/20261113T0236Z-report-prf611-certifier-key-and-case-reads.md` (synced). It has every fact with its file and line. Trees: `main` `636d15d041b5`; the onsite line as #9375's tip `34c5cbb3705c`, whose certifier, gate, record and demo files are byte for byte the base `bafae2eaec2e`'s [ran]; #9369 at `08740c9dfa46`.

**Read 1, what you can post on PRF-611**

- **Command** [read]: `warden_serve.py keys --dir KEYDIR --height 20`, from the gate's pinned tree, on the certifier's host, as `auditor`. It prints `public.json`. Both modules have the verb on all three trees, but on #9369 only `warden_serve.py keys` also makes the key's device state, so it is the one that is right everywhere.
- **Height: 20** [counted]. The count is `heads + 5 × windows` on `main` and the onsite line, `heads + 1 + 5 × windows` on #9369.
  - The demo's configuration states no period, tick or height for the certifier, and the demo has no run of record [read]. As built, a served window is one run of one 3.2 s link-window: 5 signatures (6 on #9369), under a fresh H = 10 key.
  - Certified without a break at 3.2 s: a week is 945,000 signatures (H = 20, 90.1% of the key), a month 4,050,000 (H = 22), a year 49,275,000 (H = 26). At 60 s: 50,400, 216,000 and 2,628,000 (H = 16, 18, 22).
  - H = 20 holds 7.77 days at 3.2 s, 145.6 days at 60 s. The gate takes H from 10 to 20 only, so a month at 3.2 s is past what it takes today.
  - A key that runs out is **refused, not rolled** [read]: each later link-window is listed `refused`, gets no certificate, and the run's check fails on it.
- **Time to make** [read, not run]: no kept run times it. The tree says about two minutes on eight cores, and 256 s by the gate's allowance.
- **The hash** [read]: `public.json` is `{params, public, key_id}` and has no hash field. By the challenger's convention it is the SHA-512 of the 80 public bytes: `sha512sum KEYDIR/public`.
- **Where it is read today** [read]: nowhere. `rehearse` makes its own key in a temporary directory and removes it; the gate's unit starts it as `auditor` with no `--key`. The serving node passes no signing key.
- **Directory** [read]: the tree names none. It must be under the gate's state directory, the unit's one writable path besides the pinned trees, since every signature writes the leaf state.

**Four things that would bite**

1. Nothing compares a run's key with a pinned hash. The certifier's binding has only `gate`, no `key-sha512`, and both checks read the key from the run's own `deployment.json` [read]. A posted hash is read by no code yet.
2. A key made before #9369 is on the gate's tree has no device state, and #9369's `serve` refuses it [read]. Infra would make it twice, with two hashes. Either the key waits for #9369, or infra is told the first is a stand-in.
3. On `main` and the onsite line, `serve --key` on a directory with no `seed` silently makes an H = 10 key there [read]. #9369 refuses instead.
4. `--height` is ignored when the directory already holds a key [read].

**Our one change, `rehearse --key KEYDIR`** [argued, not built]: `warden_serve.py` (about 12 lines) and the gate (about 30), three new tests in their two test files, no unit change, no Lean. The one part with a choice in it: the gate must size a request against the leaves the key has left, not 2^H. Cut it on #9369's head after its merge with `main`; before #9369 it conflicts in `rehearse` and `warden_key`.

**Read 2, the certifier's `case.json`**

- **Nothing the certifier saw is in it** [read]: no record, row, cell, salt, opened wiring or record's head.
- **Five things the certifier signed are in it** [read]: three certificates and two row registrations' certificates. Each signs a salted digest or a head of hiding roots. They are the same bytes the gate publishes in a run's `out/`.
- **The tree says two things, so I stop here.** The certifier's modules call these public. `composite.py` 19 and `served_composite.sh` 30–31 call every case's evidence and params witness. No line says which governs the certifier's case. That is soundness's.
- **Two facts vllm should have** [read]:
  - At the onsite line the case is already written to `OUT/pieces/certifier/case.json`, which is published. It is not among the files the runner moves to `private/`.
  - Its one reader is the Lean `network-period check`, run by the piece on the serving node. No composite reads it there, and the verifier's process on the auditor's verifier host reads the piece's `answer.json`, not the case.

**Not read:** which commit the gate is pinned at, whether a key directory exists on the certifier's host, and V3c's entry of the case field by field.

Ending my turn. I resume on your word that #9375 is on `main`.

