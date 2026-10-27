import Proofs.Flock.Claims.ZK.FinishRep

/-!
# The firewall's rep reads the extra claims' coordinates alone

The ring switch reads a claim's coordinates `x` and nothing else (its value and skip weights go to the verifier), and the
firewall's rep reads the extra claims only through the ring switch on `abcClaims ++ extras`. So two lists of extra
claims with the same coordinates give one rep: the firewall's `claimsOf xs` and a session's extra claims at `xs`.

* `getElem!_x`: claim `k`'s coordinates are entry `k` of the coordinates, in range or not.
* `FlockComplete.ringSwitchCut_x`: the ring switch at claims with the same coordinates is one run.
* `switchVec_x`, `ringSwitchCut_switchVec`: so is the ring switch on their switched vectors of a packed table.
* **`repFwC_claims_x`**: `repFwC` at extra claims with the same coordinates is one run.
-/

namespace FlockComplete

open Flock Transcript

/-- **Claim `k`'s coordinates are entry `k` of the coordinates**, in range or not. -/
theorem getElem!_x (cs : Array Claim) (k : ℕ) : cs[k]!.x = (cs.map (·.x))[k]! := by
  by_cases hk : k < cs.size
  · rw [getElem!_pos cs k hk, getElem!_pos (cs.map (·.x)) k (by simpa using hk), Array.getElem_map]
  · rw [getElem!_neg cs k hk, getElem!_neg (cs.map (·.x)) k (by simpa using hk)]
    rfl

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The ring switch reads the claims' coordinates alone**: at claims with the same coordinates it is one run. -/
theorem ringSwitchCut_x (io : Asks M) (sv : ℕ → Array F128) {cs cs' : Array Claim}
    (hx : cs.map (·.x) = cs'.map (·.x)) : ringSwitchCut io sv cs = ringSwitchCut io sv cs' := by
  have hs : ∀ c : Array Claim, c.size = (c.map (·.x)).size := fun c => (Array.size_map ..).symm
  unfold ringSwitchCut
  simp only [getElem!_x cs, getElem!_x cs', hs cs, hs cs', hx]

end FlockComplete

namespace Flock.Firewall.Finish

open Flock

/-- **A claim's switched vector reads its coordinates alone.** -/
theorem switchVec_x (g : Array F128) {c c' : Claim} (hx : c.x = c'.x) : switchVec g c = switchVec g c' := by
  unfold switchVec
  rw [hx]

section

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The ring switch on a packed table's switched vectors** at claims with the same coordinates is one run. -/
theorem ringSwitchCut_switchVec (io : FlockComplete.Asks M) (g : Array F128) {cs cs' : Array Claim}
    (hx : cs.map (·.x) = cs'.map (·.x)) :
    FlockComplete.ringSwitchCut io (fun k => switchVec g cs[k]!) cs =
      FlockComplete.ringSwitchCut io (fun k => switchVec g cs'[k]!) cs' := by
  have e : (fun k : ℕ => switchVec g cs[k]!) = fun k : ℕ => switchVec g cs'[k]! := by
    funext k
    refine switchVec_x g ?_
    rw [FlockComplete.getElem!_x cs, FlockComplete.getElem!_x cs', hx]
  rw [e]
  exact FlockComplete.ringSwitchCut_x io _ hx

/-- **`repFwC_claims_x`: the firewall's rep reads the extra claims' coordinates alone**: at extra claims with the same
coordinates it is one run, so at `claimsOf xs` and at any claims at `xs`. -/
theorem repFwC_claims_x (io : FlockComplete.Asks M) (hs : Hashes M) (st : Setup) (sch : Flock.Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (g : Array F128) (cm0 : FlockComplete.Commit) {E E' : Array Claim}
    (hx : E.map (·.x) = E'.map (·.x)) (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit)
    (vals : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) :
    repFwC io hs st sch tPad rep dom g cm0 E pads ic vals extra lanePad salt =
      repFwC io hs st sch tPad rep dom g cm0 E' pads ic vals extra lanePad salt := by
  unfold repFwC
  refine bind_congr fun ⟨zp, zc⟩ => ?_
  refine bind_congr fun ⟨lp, lc⟩ => ?_
  try dsimp only
  rw [ringSwitchCut_switchVec io g (cs := abcClaims st.m st.kLog zc lc ++ E)
    (cs' := abcClaims st.m st.kLog zc lc ++ E') (by rw [Array.map_append, Array.map_append, hx])]

/-- `claimsOf xs` has the coordinates `xs`. -/
theorem claimsOf_map_x (xs : Array (Array F128)) : (claimsOf xs).map (·.x) = xs := by
  unfold claimsOf
  rw [Array.map_map]
  exact Array.map_id xs

end

end Flock.Firewall.Finish
