import Proofs.Pouw.Prob
import Definitions.Claims

/-!
# S7: an event of the network audit's own coins below a sampled run's `PrBad`

The probability step of the network certifier's per-window bound (`Verity.Subprotocol.NetworkWindowLe`), with nothing of the network audit in it. The
network audit's own coin `x : α` gives a run of the sampled verifier, its tape and coins `g x`; when `g` carries the
uniform coin to the uniform tape and coins (`Pouw.pr (q ∘ g) = Claims.Pr q`), an event `A` of the coin whose every
point leads to a run in the event `B` at every extractor coin is no likelier than `B` over the tape, the run's coins
and the extractor's coins (`pr_le_of_coins`). `Claims.Pr` and `Pouw.pr` are one definition, so the marginal of `B`
over the extractor's coins is `pr_fst_eq`.
-/

namespace Verity.Network

open Finset

/-- An event that reads only the first component has its marginal's probability, the second component's type being
inhabited. -/
theorem pr_fst_eq {α β : Type} [Fintype α] [Fintype β] [Nonempty β] (P : α → Prop) :
    Pouw.pr (fun x : α × β => P x.1) = Pouw.pr P := by
  classical
  unfold Pouw.pr
  have hf : (univ.filter fun x : α × β => P x.1) = (univ.filter P) ×ˢ (univ : Finset β) := by
    ext x; simp
  rw [hf, Finset.card_product, Fintype.card_prod, Finset.card_univ]
  push_cast
  rw [mul_div_mul_right _ _ (by exact_mod_cast Fintype.card_pos.ne')]

/-- **S7: an event of the coin below the run's event, through a uniform coin map.** If the coin map `g` carries the
uniform `x : α` to the uniform tape and coins (`hg`), and every `x` in `A` leads to a tape and coins `g x` at which
every extractor coin `c` is in `B`, then `A` is no likelier than `B` over the tape, the coins and the extractor's
coins. -/
theorem pr_le_of_coins {α Ω R C : Type} [Fintype α] [Fintype Ω] [Fintype R] [Fintype C] [Nonempty C]
    (g : α → Ω × R) (hg : ∀ q : Ω × R → Prop, Pouw.pr (fun x => q (g x)) = Claims.Pr q) {A : α → Prop}
    {B : Ω × R × C → Prop} (h : ∀ x, A x → ∀ c, B ((g x).1, (g x).2, c)) : Pouw.pr A ≤ Claims.Pr B := by
  let q : Ω × R → Prop := fun y => ∀ c, B (y.1, y.2, c)
  have hm : Pouw.pr (fun s : Ω × R × C => q (s.1, s.2.1)) = Claims.Pr q := by
    rw [← Pouw.pr_equiv (Equiv.prodAssoc Ω R C)]
    exact pr_fst_eq q
  calc Pouw.pr A ≤ Pouw.pr (fun x => q (g x)) := Pouw.pr_mono h
    _ = Claims.Pr q := hg q
    _ = Pouw.pr (fun s : Ω × R × C => q (s.1, s.2.1)) := hm.symm
    _ ≤ Claims.Pr B := Pouw.pr_mono fun s hs => hs s.2.2

end Verity.Network
