# Catalog: PoUW implementations and design files (origin/main, read 29 Oct 00:10Z)

| Design | Files | Committed | Noise | γ / overhead | Floor / forced share | Perplexity | Served in vLLM |
|---|---|---|---|---|---|---|---|
| NCP `ncp-v1` (+`-shift24`) | verity/core/protocols/pouw/schemes/ncp.py, PROTOCOL.md, circuit/ncp2.py | int7 A rows; route-U tile digests; weight Y / c₀ | SHAKE E₁ / Fᵢ; 3 stacked int7 products | modelled, W1 / 4090: γ₀ = 0.5%, γ ≈ 0.5–0.9%, ~3× plain (benchmarks/pouw/gamma.py) | the γ floor from the free final / unchecked words | — | yes (`ncp-v1`, `ncp-v2`) |
| Pearl-C `pearl-c-{h100,sm120}-v1[-h3]` | schemes/pearl_c*.py, pearl_c_device.py; benches pearl_c_sm90/, pearl_c_sm120/, pearl_c_vllm/ | FP8 A and B trees, tile leaves, peel / cleanup | salt seeds + E/F lines; TT_OUT ρ | modelled credit (1 − ρ): H100 / H200 ρ = 1/400; sm_120 ρ = 1/1000. Panel: measured slowdown against stock FP8 (sm_120 end to end; sm_90 GLM path) | tile debit ≤ ρ (census rows 4ρ) | — | yes (the `-h3` device path) |
| Pearl KW `pearl-fp8-v4` | schemes/pearl_kw.py | Pearl certificate-v4 | Pearl's BLAKE3 noise | modelled, W1: no γ₀; Ω ≈ 1.00–1.14 (gamma.py) | γ floor ~0.3–8% | — | no (upstream Pearl) |
| Pearl-C4 `pearl-c-nvfp4-v0` | schemes/pearl_c4.py, pearl_c4/ | NVFP4 product + forming | an E2M1 noise atom | no γ₀ in the certificate; sm_120 only | — | — | no |
| H1 + fx1 `eps1-sm120-h1fx1` | circuit/h1tile.py, segment.py; vLLM …/pouw/h1/ | s8 A and B, admit; CP-W receipt words; exact z | 1-bit dither + U ±64; fx1 fold keys | modelled as design `dither-c-peel-fx1`: γ 2.14 (measured) / 2.28 (sheet) on the RTX PRO 6000 | ε_skip 0.36% / 0.42%; headline ε 0.54% / 0.59% | arm `co32c` | yes (eager int8_exact) |
| J10-R (`PouwH1Tile5/6`) | h1tile.py J10-R; served_h1/ | the same tile; a call-scale seed | call-seed noise (blocks once a call) | the same H1 family; no registered scheme ("No scheme takes v5 or v6") | — | — | window support only, not a PoUW scheme id |
| Dither-C peel fx1 (the sm_120 headline) | machine_model/data/designs/dither-c-peel-fx1.toml | co32c limbs + peel PV + fx1 words | co32c / cohalf / peel | modelled γ 2.136 (VF3r-16) / 2.283 (DS-120) | the ♭ floor; ε 0.540% measured / 0.585% sheet | `co32c` | via H1 only |
| Dither-2S LI-CP-W p=1 (the H200 headline) | …/sm_90/designs/dither-2s-licpw-redux-p1-l32.toml | two-sided words; LI-CP-W checkpoint | two-sided {0,1}; ffill uncredited | modelled γ 2.344 (DC-90) / 2.406 (DS-90), H200 | ε_skip 0.59% / 0.86%; headline ε 1.022% measured (over 1%) / 0.860% sheet | — | no |
| The other Dither registry designs | data/designs/*.toml, devices/sm_{80,89,90,100}/designs/ | per noise table | 1S / 2S / C × L / fold | sm_120 γ ≈ 1.9–3.2; H200 duo γ ≈ 1.0–4.6 by L (`machine-model eps --table`) | ε_skip in each `[skip.served]` | arms `adopted`, `adopted-bd-lm`, `co32c` | no |
| Dither v3.2 no-fold | verity/…/pouw/dither/PROTOCOL.md; examples/dither_full_util/ | 16-bit prefix sums + O; hm96 / sha512 rows | dithers + HYB U/V ±80 | modelled static: sm_120 γ 1.05–1.15 (pinned B, L = 1,024); H200 γ 1.40–2.15 (hash-bound); runs r20261027-031621-5469, …-3a37 | ε_skip a period; the γ ≤ 10 class rule | — | no |
| Forced share / floors | filler/share.py; data/floors.csv; ScalarRoute in the headline | — | — | a capacity share, not a scheme's γ | forced share 0.99τ; π₃₂ / ½ or the 1.22-slot ♭ | — | n/a |
| FPX candidates | benchmarks/pouw/fpx_design/ | an int8 base + FP plain / fold_add | g1 / 2SP / … | priced on sm_120 / 90 / 100 by fpx_eps.py | the V-x0 credit floor | — | no |
| Archived Pearl BLAKE3 | /workspace/archive/pouw-blake3/ | -h1 / -h2 / BLAKE3-h3 trees | Pearl BLAKE3 | frozen after the SHA-512 ruling | — | — | no |

## Card tables

- **The default is sm_120:** benchmarks/pouw/machine_model/data/ (theta.toml, designs.toml, vectors/, table.toml).
- **The H200 (sm_90):** …/data/devices/sm_90/ (model.toml is currently DC-90; vectors DS-90, DC-90 and DC-90-ri;
  theta.toml for the "H200 SXM 141 GB"; designs + honest + sheet).
- Also sm_100 (B200), sm_89 (L40S) and sm_80 (A100).

## What vLLM serves today

`integrations/vllm/…/pouw/__init__.py`, `EXECUTED`: NCP, Pearl-C's `-h3` device schemes, and `eps1-sm120-h1fx1`. The Dither
registry designs and Dither v3.2 are not served.
