# REDTEAM_V2 URGENT — accepted forgery against Candidate A (GKR v2) `run-vu`

**Date:** 2026-09-22 ~09:45Z · **Lane:** red-team-v2 · **Pod:** vy-cpu2 · **Verifier:** `verity-gkr run-vu`
(commit on `lane/red-team-v2`, `main`=4ddb170) · run `r20260922-094212-465e`.

## What was accepted

A 2-VU Goldilocks v2 bundle (`fixtures/redteam-v2/gkr/neg/chain_spec_stripped/`) in which **VU0's epilogue reads a
non-final unit's state** (unit 0's outgoing `(s,e,M,z)`), publishing `y16 = cast(unit0 state)` instead of
`cast(unit95 state)`. The real verifier **ACCEPTS** it, under both `--clear` and PCS:

```
chain_spec_stripped (clear): verify=Ok(())
chain_spec_stripped (pcs)  : verify=Ok(())
```

The honest control (same rows, honest `chain.txt`) is the `epilogue_wrong_unit` case, which is **REJECTED**
(`chain: VU 0 epilogue input (column 1) differs from the last unit's value`). The only difference between the two
bundles is the `chain.txt` file.

## Root cause — the verifier trusts the proof bundle's `chain.txt`

`backends/gkr/src/main.rs::load_vu` builds the chain linkage from the **bundle-supplied** `chain.txt`:

```
let limbs: Vec<Link> = if let Some(text) = files.chain.as_ref().and_then(|p| std::fs::read_to_string(p).ok()) {
    read_chain(&text, &unit.circ, &epi.circ)   // <-- state/ovf/sgn links come from the untrusted dir
} else { ... default word links ... };
let chain = Chain { steps, limbs, epi_y16: ..., public_y16: pcols };
```

Only the **public-word link** (`epi_y16 == public_y16`) is hardcoded in `Chain`. The four state links
`(s,e,M,z)`, the two `ovf` links and the two `sgn` links are read from `chain.txt`. A prover who ships an **empty**
`chain.txt` drops every state/ovf/sgn link; the verifier then never checks that the epilogue's input state equals the
last unit's output. The hardcoded word link still passes because we set `public = epi.y16`. `steps` is likewise
derived from the witness header (`n_units / n_pub`), not from a fixed statement.

Net: the arithmetization and sumcheck are sound (every per-row assertion and lookup still holds; see the other cases
below), but the **VU-chain statement is configured by the caller**, so the published word need not be the chain's
final state.

## Severity / scope

- **HIGH as a verifier-API / deployment trust-boundary bug.** It is not a break of the GKR sumcheck or the checker-v2
  arithmetization; it is that `run-vu` accepts caller-supplied linkage and geometry. Any deployment that hands the
  proof bundle (including `chain.txt`) to the verifier as the source of the statement is forgeable.
- The same pattern (`steps` from the header) means the batch geometry is also caller-controlled.

## Proposed fix (owning lane: a-gkr / a-verifier — NOT fixed here)

The verifier must derive the chain spec (state/ovf/sgn links, `steps`, `B`) from the **fixed circuit / statement**,
exactly as it already does for `epi_y16`/`public_y16`, rather than reading `chain.txt` from the proof directory. If a
serialized `chain.txt` must accompany the bundle, absorb it into the Fiat–Shamir transcript **and** assert it equals
the verifier's own expected spec before use. A one-line harness guard (`assert!(!limbs.is_empty() && limbs == expected)`)
would close the demonstrated instance but the principled fix is verifier-owned linkage.

## Reproduce

```
BIN=$CARGO_TARGET_DIR/release/verity-gkr
$BIN run-vu --dir fixtures/redteam-v2/gkr/neg/chain_spec_stripped --clear   # verify=Ok(())
$BIN run-vu --dir fixtures/redteam-v2/gkr/neg/chain_spec_stripped           # verify=Ok(()) (pcs)
```

Regenerate the fixture: `python -m verity_numerical.redteam.v2_forge gkr-chain --out fixtures/redteam-v2/gkr`.
