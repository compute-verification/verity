import Definitions.Hash.Keyed
import Definitions.Hash.Hm96
import Definitions.Pouw.Basic.Prob

/-!
# The keyed leaf's hiding game (trusted)

A window's lead commits each of its `n` instances (a call's `A` rows, a forward's words) as a keyed leaf
(`Hash.Keyed`): instance `i`'s public digest is `H(ks ‖ salt ‖ top i)` over its top message `top i`, under the window's
key `ks` (64 OS bytes) and live salt `salt`. Before the salt is drawn it records the key leaf, the hm96 leaf over `ks`
under 192 OS salt bytes `y₀` (`Hm96.leaf` at `x = ks`). What the verifier sees of the window is the key leaf and the
digests (`view`), and `real` is the probability that a predicate `A` of the view holds, over a uniform `ks` and `y₀`.

Hiding is that two windows' views are indistinguishable. Each step from a window's view to one that no longer reads its
tops is an explicit distinguisher's advantage, built from `A`, in a game of its own:

* `hideAdvantage`: hm96's hiding game, the salt's mask `M·y₀` against a fresh uniform 64 bytes, beside the salt's
  digest `c = H(saltPrefix ‖ y₀)`, with independent coins `ω` the distinguisher reads. `hideDistinguisher` is the one
  that reads the key leaf from the mask (or the uniform bytes) xored with `ks`, and the digests from `ks`.
* `prfAdvantage`: the keyed digest `ks ↦ (H(ks ‖ salt ‖ top i))_i` of a uniform key against a random function of the
  top messages (`fun i => R (rep top i)`, so equal messages get one answer), with independent coins `ω`.
  `prfDistinguisher` is the one that reads the key leaf from a uniform commit string, which the hiding step left
  independent of `ks`.

`advantage` is their sum, each taken in absolute value. Once the mask is replaced, the key leaf is a function of
uniform bytes independent of `ks`, and once the digests are, a view of distinct top messages is a uniform tuple
whatever they are.
-/

namespace Pouw.KeyedLeaf

open Hash

/-- The key leaf's 192 OS salt bytes. -/
abbrev KeySalt := Fin Hm96.SALT_BYTES → UInt8

/-- The instances' top messages, each under its own domain. -/
def tops {n : ℕ} (dom : Fin n → Tree.Domain) (t : Fin n → Keyed.Top) : Fin n → Msg :=
  fun i => (t i).msg (dom i)

/-- What the verifier sees of a window keyed by `ks` with the key leaf's salt `y₀`: the key leaf, and each instance's
keyed digest. -/
def view {n : ℕ} (H : Msg → Digest) (key : Msg) (salt : Digest) (top : Fin n → Msg) (ks : Digest) (y₀ : KeySalt) :
    Digest × (Fin n → Digest) :=
  (Hm96.leaf H key ks (List.ofFn y₀), fun i => Keyed.digest H ks salt (top i))

/-- The probability that `A` holds of the view, over a uniform key and key-leaf salt. -/
noncomputable def real {n : ℕ} (H : Msg → Digest) (key : Msg) (salt : Digest) (top : Fin n → Msg)
    (A : Digest × (Fin n → Digest) → Prop) : ℝ :=
  pr fun p : Digest × KeySalt => A (view H key salt top p.1 p.2)

/-- The salt's mask `M·y₀` under the verifier's key, as 64 bytes. -/
def mask (key : Msg) (y₀ : KeySalt) : Digest := fun i => (Hm96.mask key (List.ofFn y₀)).getD i 0

/-- The salt's digest `c = H(saltPrefix ‖ y₀)`. -/
def saltDigest (H : Msg → Digest) (y₀ : KeySalt) : Digest := H (Hm96.saltPrefix ++ List.ofFn y₀)

/-- The hm96 leaf over the commit string `b ‖ c`. -/
def leafOf (H : Msg → Digest) (key : Msg) (b c : Digest) : Digest := H (Hm96.leafPrefix H key ++ (b.bytes ++ c.bytes))

/-- hm96's hiding game for a distinguisher `E` with independent coins `ω`: `E` of the mask against `E` of a uniform
64 bytes, each beside the salt's digest. -/
noncomputable def hideAdvantage {ω : Type} [Fintype ω] (H : Msg → Digest) (key : Msg)
    (E : ω × (Digest × Digest) → Prop) : ℝ :=
  pr (fun p : ω × KeySalt => E (p.1, (mask key p.2, saltDigest H p.2))) -
    pr (fun p : ω × (Digest × KeySalt) => E (p.1, (p.2.1, saltDigest H p.2.2)))

/-- The least index whose top message is instance `i`'s. -/
def rep {n : ℕ} (top : Fin n → Msg) (i : Fin n) : Fin n :=
  (Finset.univ.filter fun j => top j = top i).min' ⟨i, Finset.mem_filter.2 ⟨Finset.mem_univ i, rfl⟩⟩

/-- The keyed digest's game for a distinguisher `D` with independent coins `ω`: `D` of the digests under a uniform key
against `D` of a uniform function of the top messages. -/
noncomputable def prfAdvantage {ω : Type} [Fintype ω] {n : ℕ} (H : Msg → Digest) (salt : Digest) (top : Fin n → Msg)
    (D : ω × (Fin n → Digest) → Prop) : ℝ :=
  pr (fun p : ω × Digest => D (p.1, fun i => Keyed.digest H p.2 salt (top i))) -
    pr (fun p : ω × (Fin n → Digest) => D (p.1, fun i => p.2 (rep top i)))

/-- `A`'s distinguisher in the hiding game: the key `ks` is its coins, and the key leaf is over `ks` xored with the
mask (or the uniform bytes) and the salt's digest. -/
def hideDistinguisher {n : ℕ} (H : Msg → Digest) (key : Msg) (salt : Digest) (top : Fin n → Msg)
    (A : Digest × (Fin n → Digest) → Prop) : Digest × (Digest × Digest) → Prop :=
  fun p => A (leafOf H key (fun i => p.1 i ^^^ p.2.1 i) p.2.2, fun i => Keyed.digest H p.1 salt (top i))

/-- `A`'s distinguisher in the keyed digest's game: its coins are a uniform 64 bytes and the key leaf's salt, and the
key leaf is over them. -/
def prfDistinguisher {n : ℕ} (H : Msg → Digest) (key : Msg) (A : Digest × (Fin n → Digest) → Prop) :
    (Digest × KeySalt) × (Fin n → Digest) → Prop :=
  fun p => A (leafOf H key p.1.1 (saltDigest H p.1.2), p.2)

/-- `A`'s advantage against a window of top messages `top`: its hiding distinguisher's and its keyed-digest
distinguisher's, each in absolute value. -/
noncomputable def advantage {n : ℕ} (H : Msg → Digest) (key : Msg) (salt : Digest) (top : Fin n → Msg)
    (A : Digest × (Fin n → Digest) → Prop) : ℝ :=
  |hideAdvantage H key (hideDistinguisher H key salt top A)| + |prfAdvantage H salt top (prfDistinguisher H key A)|

end Pouw.KeyedLeaf
