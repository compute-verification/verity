[2026-11-12T22:45Z] c0a77b5dac0a
SCANCOUNT to network-accounting (lead), Thu 12 Nov 2:45 PM PST. **NET-401's read, nothing built: every gate the scan's layout names is a template instance's parameter or return, and no position needs a file's rows. Two things to rule before building: `inBit` and `outBit`, run as code, lay out every unit; and one clause of the fit needs one pass over each ingress file's rows.**

Bits is `verity/protocols/spot/verifier/lean/Spot/Bits.lean`.

**1. The gates**
- Egress presence and frame: the egress instance's return (`security/Properties/NetworkScan.lean:86-88`), so `outBit` [read].
- Ingress: the `row` parameter (`:89-91`), so `inBit` [read].
- The view: the egress `state` parameter (`NetworkScan/Chain.lean:98-100`), so `inBit` [read]. `sin`, the entries and the returned state and view likewise (`Chain.lean:86-106`, `190-195`).
- `Bits.check` reads a file's header only (`Bits:773-824`) [read].
- **Run as code, `inBit` and `outBit` build the member's key program, every unit** (`Bits:564`, `568`, `237-251`), a form `tests/test_verifier_cost.py:131` refuses. `scanOf` needs no position, only `Bits.check`; `scanLayout` is never compiled [argued].
- **Needs the rows: that an ingress `row` bit is an input gate**, which the fit asks (`security/Definitions/NetworkCertifier/SampledAudit.lean:132`). In Bits' layout a source gate no row reads is the constant 0 (`Bits:597-599`). `unitReads` (`Bits:830-833`) decides it, one pass a file, with `readsWhy` (`:839-857`) [read]. Nothing calls either [read, by search].

**2. The require: yes.** `[[require]] name = "core_spot"`, `path = "../../../spot/verifier/lean"`. Spot's package is already in the closure through `nci_check` (the certifier's `lakefile.toml:14-16`, NCI's `:28-30`) [read]. `tests/test_lean_packages.py` has no rule on it (`:122`, `:248`, `:256`, `:264`); `tests/test_boundaries.py:721` forbids only Spot importing the certifier [read]. Not built.

**3. Order, in declarations** (existing counts [read], the rest [argued])
- A. `ScopeMatches` out: 6 changed.
- B. `scanOf` holds the three files to their templates: about 8 of code, `Accept.lean`'s 15 extended. The period's input gains the files; it holds none (`Period.lean:359-426`) [read].
- C. Statements and proofs over Bits' layout. `StmtFlattens` and `Flattens` leave here: 5 statements restated, 68 retire (`Walk`, `Causal`, `Fits`), about 40 new; then NET-285's 57.
- **I name no date: I can size the work, not time it.** It rests on #9375 on `main`, your ruling on the rows pass, the scan's real files passing `Bits.check` (nothing has run it), soundness's review of C, and pool slots.

**Not read:** the scan's real circuit files; what fixes a member's unit order (bucket `t` as unit `t`); how the files reach the period check; the bridge lemmas' hypotheses at the scan's files; NCI's record proofs.

**The inbox against this turn:** nothing differs in what stands. The 1:56 PM text's one page, its Mon 16 Nov ask to soundness and its ask to proofs for a lemma are replaced by the 2:06 PM text, as this turn says.

**The sibling stop:** no try of mine needs it before #9375 lands, so I did not run lean's cure.

[2026-11-12T22:47Z] 03ad95d00d8c
WATERLINE to the network-accounting lead, Thu 12 Nov 2:47 PM PST.

**3. A capture was taken at `-s 54` today and the reader read it. I launched no run: the one you describe exists.**

- `r20261112-174228-bacb` was taken at 130 and cut afterwards [read: its source's script].
- **`r20261112-193846-84a6`**, 11:38 AM PST, at #9428's head, source clean, exit 0 [read: its record]. No log line holds the command; the script's line 51 at that commit does: `tcpdump -i lo -U -s 54 -w "$W/lo.pcap" tcp` [read]. The reader gives link type 1 (Ethernet), snapshot 54, "agrees" both ways, 201 of 201 pairs [ran].
- Taken on `lo`, not `any` [read].
- **Not shown:** the README's own command, which filters while capturing. Its filter was applied to the kept file [ran]. One run, if you want it.

**What 54 cuts on the serving node:**

- **IPv4 with 8 or more bytes of IP options** (56): refused by name [ran, built packets]. Ordinary sockets send none [argued]; the node's, I do not know.
- **IPv6** (68): never in the file, since `ip host` keeps IPv4 alone [ran, built packets]. If the forward reaches the server over IPv6's loopback, the file misses the run and nothing is refused. The forward's far end is the launch's [read]; tonight's, I do not know.
- **A cooked capture** (`any`): read at 54. v1 needs 50, v2 exactly 54 [ran, built packets].

**Rulings 1 and 2:** head `d23978eac71eb08706943416a7f93eeabedf70c9`, 54 tests pass [ran]. The new name is "shorter than the headers it states". Its line is in the reader's docstring list: the README has no list of refusals [read], so it has no diff. Say if you want one there.

4 is not started.

TIMER: none

