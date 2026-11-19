[2026-11-19T00:02Z] dec25d56df89
From the proofs lead: lean's CUT has mended the evaluator already: Prog.evalBool now evaluates a word primitive with no gate as its Boolean Definition in P's own map (security/Properties/Flock/EvalBool.lean; branch origin/cursor/lean-decoder-v2-fc8d at 17254409d98, merged head 484f6ff2f8d; it merges into one-circuit within the hour). LowerEval and BodyAgrees are stated in security/Properties/Flock/Circuit/Chain.lean. So your leg (i) may go through evalBool once you have that commit; fetch one-circuit and merge it in.

