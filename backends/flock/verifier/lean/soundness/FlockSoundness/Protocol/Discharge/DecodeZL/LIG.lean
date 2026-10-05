import FlockSoundness.Protocol.Discharge.DecodeZL.Statements

/-!
# Ligerito's rounds carry its messages (R11b)

`lig_carries`: Ligerito's frames `ligFrames`, read by `wLIG`, are `ligMsgs`, cut at `ligShape`'s coins. `Carries` reads
the spine only through its coin pattern, so the rounds are proved at a pattern (`CarriesP`) and glued with
`CarriesP.append`: the start, the first out-of-domain value (`carriesP_f`), the `k0` folding rounds (`carriesP_fold`),
the last one with the level-1 cap (`carriesP_cap`), level 0's out-of-domain samples and queries (`carriesP_mid`), and
the level loop (`carriesP_levels`, by recursion on the levels left, ending at `yr` with `carriesP_yr`). `fast100`'s
schedules fold 4 lanes above level 0, so `ligShape`'s pattern is the rounds' (`isCoin_rec`), and their caps and query
counts fit the frames' bounds (`fast100_small`).
-/

namespace FlockSoundness.Discharge.DecodeZL

open Flock Refine Exec Game Model

/-! ## Carrying at a coin pattern -/

/-! ## Ligerito's rounds -/

/-- The wires of a level's two out-of-domain rounds (the second point's coins carry nothing), its two query rounds and
its query message. -/
def wMid : List (List Wire) := [[.f, .kk], [], [.f, .kk], [], [], [.kk]]

/-- **The level loop's wires**, `j` levels left: the first fold coin's round carries nothing, three fold rounds their
message, the last fold round its message and the next level's cap, then `wMid`; at the last level, the last fold round's
message and `yr`, then the two query rounds and the batching coin's carry nothing. -/
noncomputable def wLev (ext : ℕ) : ℕ → List (List Wire)
  | 0 => []
  | 1 => [[]] ++ (List.replicate 3 [.kk] ++ ([[.kk, .v (2 ^ (ext + 1))]] ++ ([[]] ++ [[]])))
  | j + 2 => [[]] ++ (List.replicate 3 [.kk] ++ ([[.kk, .cap]] ++ (wMid ++ wLev ext (j + 1))))

/-! ## The coin pattern of Ligerito's spine -/

/-- **The Ligerito wires**: the start's round carries nothing, the first out-of-domain round its value, the `k0` fold
rounds their messages, the last one also the level-1 cap, then level 0's `wMid` and the level loop's `wLev`. -/
noncomputable def wLIG (s : Accounting.Schedule) : List (List Wire) :=
  [[]] ++ ([[.f]] ++ (List.replicate s.k0 [.kk] ++ ([[.kk, .cap]] ++ (wMid ++ wLev s.ext (s.levels.length - 1)))))

end FlockSoundness.Discharge.DecodeZL
