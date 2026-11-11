module

public import Identity.Domains.Shared

/-!
PoUW's domain separators, with Pearl's own labels: the entries of the one table (`Identity.Domains`, whose docstring has
its rules) that only PoUW reads. PoUW's Lean imports this module, and through it the shared entries it reads
(`Identity.Domains.Shared`), apart from the other protocols' separators.
-/

@[expose] public section

namespace Domains

-- PoUW: the protocol and its identifier
abbrev POUW_PROTOCOL : String := "verity/pouw/v1"
abbrev POUW_IDENTIFIER : String := "verity/pouw/identifier/v1"
abbrev POUW_PASS_MANIFEST : String := "verity/pouw/pass-manifest/v1"

-- PoUW: epochs: coins, salts and the weights they bind
abbrev POUW_EPOCH_COINS : String := "verity/pouw/epoch-coins/v1"
abbrev POUW_EPOCH_COINS_RECORDED : String := "verity/pouw/epoch-coins/v2"
abbrev POUW_EPOCH_COINS_LIVE : String := "verity/pouw/epoch-coins/live/v1"
abbrev POUW_EPOCH_COIN : String := "verity/pouw/epoch-coin/v1"
abbrev POUW_EPOCH_SALT : String := "verity/pouw/epoch-salt/v1"
abbrev POUW_UNVERIFIED_SALT : String := "verity/pouw/salt/v1"
abbrev POUW_EPOCH_OPEN : String := "verity/pouw-epoch-open/v1"
abbrev POUW_ANCHORED_EPOCH_OPEN : String := "verity/pouw-epoch-open/v2"
abbrev POUW_WEIGHTS_ROOT : String := "verity/pouw/weights/v1"
abbrev POUW_WEIGHTS_ROOT_RECORDED : String := "verity/pouw/weights/v2"
abbrev POUW_WEIGHT_ROWS : String := "verity/pouw/weight-rows/v1"
abbrev POUW_ACTIVATION_ROWS : String := "verity/pouw/activation-rows/v1"
abbrev POUW_TRANSCRIPT : String := "verity/pouw/transcript/v1"
abbrev POUW_AUDIT_ROW : String := "verity/pouw/row/v1"
abbrev POUW_AUDIT_TILE : String := "verity/pouw/tile/v1"
abbrev POUW_ROTATION_KEY : String := "verity/pouw/rotation-key/v1"

-- PoUW: a period (the bytes say window): its salt, record, opening and public inputs
abbrev POUW_PERIOD_SALT : String := "verity/pouw-salt/v1"
abbrev POUW_RECORD : String := "verity/pouw-record/v1"
abbrev POUW_PROVER : String := "verity/pouw-prover/v1"
abbrev POUW_PERIOD_OPEN : String := "verity/pouw-window-open/v1"
abbrev POUW_PERIOD_VALUES : String := "verity/pouw-window-values/v1"
abbrev POUW_PERIOD_RECEIPT : String := "verity/pouw/window-receipt/v1"
abbrev POUW_PERIOD_T : String := "verity/pouw/window/t/v1"
abbrev POUW_PERIOD_ROOT : String := "verity/pouw/window/root/v1"
abbrev POUW_PERIOD_DOMAIN : String := "verity/pouw/window/domain/v1"
abbrev POUW_PERIOD_DOMAIN_H3 : String := "verity/pouw/window/domain-h3/v1"
abbrev POUW_PERIOD_SIDE_DOMAIN : String := "verity/pouw/window/side-domain/v1"
abbrev POUW_PERIOD_SIDE_DOMAIN_A : String := "verity/pouw/window/side-domain-a/v1"
abbrev POUW_PERIOD_SIDE_DOMAIN_B : String := "verity/pouw/window/side-domain-b/v1"
abbrev POUW_ROUND_NOISE : String := "verity/pouw/round-noise/v1"
abbrev POUW_PUBLIC_ROW_SALT : String := "verity/pouw/public-row-salt/v1"

-- PoUW: sampling (the bytes say sample)
abbrev POUW_AUDIT_KEY : String := "verity/pouw/audit/v1"
abbrev POUW_PEARL_C_SAMPLING : String := "verity/pouw/pearl-c-draw/v1"
abbrev POUW_PEARL_C_SAMPLING_AFTER : String := "verity/pouw/pearl-c/draw-after/v1"
abbrev POUW_SERVED_ZK_SAMPLING : String := "verity/pouw/served-zk-draw/v1"
abbrev POUW_CLOSURE_SAMPLING : String := "verity/pouw/closure-draw/v1"
abbrev POUW_H1_SAMPLING : String := "verity/pouw/h1/draw/v1"

-- PoUW: H1: its calls, rounds, commitment, requests and weights
abbrev POUW_H1_CALLS : String := "verity/pouw/h1/calls/v2"
abbrev POUW_H1_ROUND_A : String := "verity/pouw/h1/round-a/v1"
abbrev POUW_H1_COMMIT : String := "verity/pouw/h1/commit/v1"
abbrev POUW_H1_REQUEST_IDS : String := "verity/pouw/h1/requests/ids/v1"
abbrev POUW_H1_WEIGHTS : String := "verity/pouw/h1/weights/v1"
abbrev POUW_H1_OK_ROW_SALT : String := "verity/pouw/h1-prove/ok-row-salt/v1"

-- PoUW: noise, seeds and receipts
abbrev POUW_SEED : String := "verity/pouw/seed/v2"
abbrev POUW_RECEIPT : String := "verity/pouw/receipt/v1"
abbrev POUW_NOISE : String := "verity/pouw/noise/v1"
abbrev POUW_CALL_SEED : String := "verity/pouw/call-seed/v1"
abbrev POUW_EPS1_SEED : String := "eps1-sm120-h1fx1/seed/v1"
abbrev POUW_DITHER_CALL : String := "verity/pouw/dither/call/v1"
abbrev POUW_DITHER_FILL : String := "verity/pouw/dither/fill/v1"
abbrev POUW_DITHER_OPERAND : String := "verity/pouw/dither/operand/v1"
abbrev POUW_DITHER_CHECKPOINT : String := "verity/pouw/dither/checkpoint/v1"
abbrev POUW_DITHER_SEEDS : String := "verity/pouw/dither/seeds/v1"
abbrev POUW_DITHER_FILL_SEEDS : String := "verity/pouw/dither/fill-seeds/v1"

-- PoUW: schemes: NCP, Pearl-C's records and anchors
abbrev POUW_NCP_E1 : String := "verity/pouw/ncp/E1/v1"
abbrev POUW_NCP_F1 : String := "verity/pouw/ncp/F1/v1"
abbrev POUW_NCP_TILE : String := "verity/pouw/ncp/tile/v1"
abbrev POUW_PEARL_C_RECORDS : String := "verity/pouw/pearl-c/records/v2"
abbrev POUW_PEARL_C_SPLIT : String := "verity/pouw/pearl-c/split/v1"
abbrev POUW_ANCHOR_PEARL_C_EA : String := "verity/pouw/anchor/pearl-c/ea/v1"
abbrev POUW_ANCHOR_PEARL_C_EB : String := "verity/pouw/anchor/pearl-c/eb/v1"
abbrev POUW_ANCHOR_PEARL_C_FA : String := "verity/pouw/anchor/pearl-c/fa/v1"
abbrev POUW_ANCHOR_PEARL_C_FB : String := "verity/pouw/anchor/pearl-c/fb/v1"
abbrev POUW_ANCHOR_PEARL_C_G : String := "verity/pouw/anchor/pearl-c/g/v1"
abbrev POUW_ANCHOR_PEARL_C4_EA : String := "verity/pouw/anchor/pearl-c4/ea/v2"
abbrev POUW_ANCHOR_PEARL_C4_EB : String := "verity/pouw/anchor/pearl-c4/eb/v2"
abbrev POUW_ANCHOR_PEARL_C4_FA : String := "verity/pouw/anchor/pearl-c4/fa/v2"
abbrev POUW_ANCHOR_PEARL_C4_FB : String := "verity/pouw/anchor/pearl-c4/fb/v2"
abbrev POUW_ANCHOR_PEARL_C4_G : String := "verity/pouw/anchor/pearl-c4/g/v2"

-- PoUW: row schemas of the tile circuits' leaves
abbrev POUW_ROW_BF16 : String := "pouw/row-bf16/v1"
abbrev POUW_ROW_FP32 : String := "pouw/row-fp32/v1"
abbrev POUW_ROW_NVFP4_P : String := "pouw/row-nvfp4-p/v1"
abbrev POUW_ROW_FP8_P : String := "pouw/row-fp8-p/v1"
abbrev POUW_ROW_FP8_PI : String := "pouw/row-fp8-pi/v1"
abbrev POUW_ROW_OK : String := "pouw/row-ok/v1"
abbrev POUW_ROW_SD : String := "pouw/row-sd/v1"
abbrev POUW_ROW_PC8_HEAD : String := "pouw/row-pc8-head/v1"
abbrev POUW_ROW_PC8_CARRY : String := "pouw/row-pc8-carry/v1"
abbrev POUW_ROW_FP8_CODES : String := "pouw/row-fp8-codes/v1"
abbrev POUW_ROW_PC4_HEAD : String := "pouw/row-pc4-head/v1"
abbrev POUW_ROW_PC4_CARRY : String := "pouw/row-pc4-carry/v1"
abbrev POUW_ROW_NVFP4_CODES : String := "pouw/row-nvfp4-codes/v1"
abbrev POUW_TILE_H0 : String := "pouw/tile-h0/v1"
abbrev POUW_CAP_WORD : String := "pouw/cap-word/v1"
abbrev POUW_CAP_FLAGS : String := "pouw/cap-flags/v1"
abbrev POUW_CAP_ROWS : String := "pouw/cap-rows/v1"
abbrev POUW_CAP_CHUNKS : String := "pouw/cap-chunks/v1"
abbrev POUW_CAP_SUMS : String := "pouw/cap-sums/v1"

-- Pearl's own labels, as Pearl defines them (never renamed)
abbrev PEARL_C_SEED_A : String := "pearl-c/v0/seed-A"
abbrev PEARL_C_SEED_B : String := "pearl-c/v0/seed-B"
abbrev PEARL_C4_SEED_A : String := "pearl-c4/v1/seed-A"
abbrev PEARL_C4_SEED_B : String := "pearl-c4/v1/seed-B"
abbrev PEARL_C4_LINE : String := "pearl-c4/v2/line"

end Domains
