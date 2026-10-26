import Flock.With.SimSetup
import Proofs.Flock.Claims.Complete.CostWalk

/-!
# The setup's hash calls, counted function by function, in any hash

The counting oracle (`Hashes.counted h`) runs a `With` function and counts its SHA-512 calls. `countC` is the cost
bound on it (`FlockComplete.Cost`): from any count, a run adds at most `n` calls. Each function the setup calls a
bounded number of times is bounded here, for every `h`: the digests (`partitionDigestWith`, `publicDigestWith`), the
decoders (`Program.decodeWith`, `Desc.parseWith`, `HmRow.parseWith`), the loads and draws of a public file, the
partition's checks and laws, and the links check. Each proof walks the definition with `countC`'s rules: the walk
assigns each bound, and `omega` closes what it leaves. The walk is spelled out in each proof rather than kept as a
macro, which only a `compile_time` module may hold.
-/

namespace RecDeploy.SetupCount

open Flock FlockComplete
open Lean (Json)

theorem go_bind {α β : Type} (x : CountV α) (f : α → CountV β) (k : Nat) :
    (x >>= f).go k = match x.go k with
      | (.ok a, k') => (f a).go k'
      | (.error e, k') => (.error e, k') := by
  simp only [StV.go, Id.run, ExceptT.run_bind, StateT.run_bind]
  rcases hr : x.run.run k with ⟨r, k'⟩
  cases r <;> rfl

/-- **The counting oracle's cost bound**: from any count, a run adds at most `n` calls. -/
def countC : Cost CountV where
  P x n := ∀ k, (x.go k).2 ≤ k + n
  mono h hnm k := by have := h k; omega
  pure _ k := Nat.le_add_right _ _
  bind {α β x f n m} hx hf k := by
    rw [go_bind]
    have h1 := hx k
    rcases hr : x.go k with ⟨r, k'⟩
    rw [hr] at h1
    cases r with
    | error e => dsimp only at h1 ⊢; omega
    | ok a => have := hf a k'; dsimp only at h1 ⊢; omega
  throw _ _ k := Nat.le_add_right _ _
  lift v k := by cases v <;> exact Nat.le_add_right _ _

theorem sha_cost (h : ByteArray → ByteArray) (b : ByteArray) : countC.P ((Hashes.counted h).sha512 b) 1 :=
  fun _ => Nat.le_refl _

/-- A `throw` at no calls, so that the walk assigns its bound. -/
theorem throw0 {α : Type} (e : String) : countC.P (throw e : CountV α) 0 := countC.throw e 0

/-! The walk's rules at a bound an earlier branch already fixed: a `bind` within it, a branch in it on both sides. -/

theorem throwBind {α β : Type} (e : String) (f : α → CountV β) :
    countC.P ((throw e : CountV α) >>= f) 0 := fun k => by
  rw [go_bind]; exact Nat.le_add_right _ _

theorem throwBindAny {α β : Type} (e : String) (f : α → CountV β) (N : Nat) :
    countC.P ((throw e : CountV α) >>= f) N := fun k => by
  rw [go_bind]; exact Nat.le_add_right _ _

theorem bindAny {α β : Type} {x : CountV α} {f : α → CountV β} {n m N : Nat} (hx : countC.P x n)
    (hf : ∀ a, countC.P (f a) m) (hN : n + m ≤ N) : countC.P (x >>= f) N :=
  countC.mono (countC.bind hx hf) hN

theorem pureAny {α : Type} (a : α) (N : Nat) : countC.P (pure a : CountV α) N :=
  countC.mono (countC.pure a) (Nat.zero_le _)

theorem liftAny {α : Type} (v : V α) (N : Nat) : countC.P (monadLift v : CountV α) N :=
  countC.mono (countC.lift v) (Nat.zero_le _)

theorem iteAny {α : Type} {c : Prop} [Decidable c] {x y : CountV α} {N : Nat} (hx : c → countC.P x N)
    (hy : ¬c → countC.P y N) : countC.P (if c then x else y) N := by
  by_cases hc : c
  · rw [ite_eq_left hc]; exact hx hc
  · rw [ite_eq_right hc]; exact hy hc

theorem diteMax {α : Type} {c : Prop} [Decidable c] {x : c → CountV α} {y : ¬c → CountV α} {n m : Nat}
    (hx : ∀ h, countC.P (x h) n) (hy : ∀ h, countC.P (y h) m) : countC.P (if h : c then x h else y h) (max n m) := by
  by_cases hc : c
  · rw [dite_eq_left hc]; exact countC.mono (hx hc) (Nat.le_max_left _ _)
  · rw [dite_eq_right hc]; exact countC.mono (hy hc) (Nat.le_max_right _ _)

theorem diteAny {α : Type} {c : Prop} [Decidable c] {x : c → CountV α} {y : ¬c → CountV α} {N : Nat}
    (hx : ∀ h, countC.P (x h) N) (hy : ∀ h, countC.P (y h) N) : countC.P (if h : c then x h else y h) N := by
  by_cases hc : c
  · rw [dite_eq_left hc]; exact hx hc
  · rw [dite_eq_right hc]; exact hy hc

/-- A loop over a range whose steps make no call. -/
theorem forIn_range0 {β : Type} (r : Std.Legacy.Range) (b : β) {f : Nat → β → CountV (ForInStep β)}
    (hf : ∀ a b, countC.P (f a b) 0) : countC.P (forIn r b f) 0 := by
  have := countC.forIn_range r b hf
  rwa [Nat.mul_zero] at this

theorem forIn_range0Any {β : Type} (r : Std.Legacy.Range) (b : β) {f : Nat → β → CountV (ForInStep β)} {N : Nat}
    (hf : ∀ a b, countC.P (f a b) 0) : countC.P (forIn r b f) N :=
  countC.mono (forIn_range0 r b hf) (Nat.zero_le _)

theorem forIn_array0Any {α β : Type} (xs : Array α) (b : β) {f : α → β → CountV (ForInStep β)} {N : Nat}
    (hf : ∀ a ∈ xs, ∀ b, countC.P (f a b) 0) : countC.P (forIn xs b f) N :=
  countC.mono (countC.forIn_array0 xs b hf) (Nat.zero_le _)

theorem forIn_list0Any {α β : Type} (l : List α) (b : β) {f : α → β → CountV (ForInStep β)} {N : Nat}
    (hf : ∀ a ∈ l, ∀ b, countC.P (f a b) 0) : countC.P (forIn l b f) N :=
  countC.mono (countC.forIn_list0 l b hf) (Nat.zero_le _)

theorem mapM_array0Any {α β : Type} (xs : Array α) {f : α → CountV β} {N : Nat} (hf : ∀ a, countC.P (f a) 0) :
    countC.P (xs.mapM f) N :=
  countC.mono (countC.mapM_array xs hf) (Nat.zero_le _)

variable (h : ByteArray → ByteArray)

theorem partitionDigest_cost (obj : Json) : countC.P (HmRow.partitionDigestWith (Hashes.counted h) obj) 1 := by
  apply countC.mono
  · unfold HmRow.partitionDigestWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem publicDigest_cost (header : Json) (pubs : ByteArray) :
    countC.P (HmRow.publicDigestWith (Hashes.counted h) header pubs) 1 := by
  apply countC.mono
  · unfold HmRow.publicDigestWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem descParse_cost (bytes : ByteArray) : countC.P (TemplateQuery.Desc.parseWith (Hashes.counted h) bytes) 1 := by
  apply countC.mono
  · unfold TemplateQuery.Desc.parseWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

set_option maxHeartbeats 2000000 in
theorem decode_cost (bytes : ByteArray) : countC.P (Program.decodeWith (Hashes.counted h).sha512 bytes) 1 := by
  apply countC.mono
  · unfold Program.decodeWith
    repeat (first
      | extract_lets
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible apply throw0)
      | (with_reducible apply forIn_range0)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible apply throw0)
        | (with_reducible apply forIn_range0)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

theorem classOf_cost (bytes : ByteArray) (head : String) (mj : Json) :
    countC.P (HmRow.classOfWith (Hashes.counted h) bytes head mj) 1 := by
  apply countC.mono
  · unfold HmRow.classOfWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem checkPartition_cost (c : Circuit) (obj : Option Json) :
    countC.P (HmRow.checkPartitionWith (Hashes.counted h) c obj) 1 := by
  unfold HmRow.checkPartitionWith
  extract_lets part rule digest
  clear_value part rule digest
  by_cases hr : rule = "verity/partition/v1"
  · subst hr
    cases obj with
    | none => simp only; exact countC.throw _ _
    | some o =>
      simp only
      apply countC.mono
      · repeat (first
          | extract_lets
          | (with_reducible apply throwBind)
          | (with_reducible apply Cost.bind)
          | (with_reducible apply bindAny)
          | (with_reducible apply partitionDigest_cost h)
          | (with_reducible intro _)
          | (with_reducible apply Cost.pure)
          | (with_reducible apply pureAny)
          | (with_reducible apply throw0)
          | (with_reducible apply Cost.throw)
          | (with_reducible apply Cost.lift)
          | (with_reducible apply liftAny)
          | (with_reducible apply Cost.forIn_array)
          | (with_reducible apply Cost.forIn_array0)
          | (with_reducible apply forIn_array0Any)
          | (with_reducible apply Cost.forIn_list)
          | (with_reducible apply Cost.forIn_list0)
          | (with_reducible apply forIn_list0Any)
          | (with_reducible apply forIn_range0)
          | (with_reducible apply forIn_range0Any)
          | (with_reducible apply Cost.mapM_array)
          | (with_reducible apply mapM_array0Any)
          | (with_reducible apply Cost.ite)
          | (with_reducible apply iteAny)
          | (with_reducible apply diteMax)
          | (with_reducible apply diteAny)
          | split)
        iterate 40
          all_goals try (dsimp +zetaDelta -zeta only)
          all_goals repeat (first
            | extract_lets
            | (with_reducible apply throwBind)
            | (with_reducible apply Cost.bind)
            | (with_reducible apply bindAny)
            | (with_reducible apply partitionDigest_cost h)
            | (with_reducible intro _)
            | (with_reducible apply Cost.pure)
            | (with_reducible apply pureAny)
            | (with_reducible apply throw0)
            | (with_reducible apply Cost.throw)
            | (with_reducible apply Cost.lift)
            | (with_reducible apply liftAny)
            | (with_reducible apply Cost.forIn_array)
            | (with_reducible apply Cost.forIn_array0)
            | (with_reducible apply forIn_array0Any)
            | (with_reducible apply Cost.forIn_list)
            | (with_reducible apply Cost.forIn_list0)
            | (with_reducible apply forIn_list0Any)
            | (with_reducible apply forIn_range0)
            | (with_reducible apply forIn_range0Any)
            | (with_reducible apply Cost.mapM_array)
            | (with_reducible apply mapM_array0Any)
            | (with_reducible apply Cost.ite)
            | (with_reducible apply iteAny)
            | (with_reducible apply diteMax)
            | (with_reducible apply diteAny)
            | split)
        all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
      · omega
  · cases obj with
    | none => simp only; exact countC.mono (countC.pure _) (Nat.zero_le _)
    | some o => simp only; exact countC.throw _ _

theorem templateStrata_cost (law : String) (obj : Json) (program : ByteArray) :
    countC.P (HmRow.templateStrataWith (Hashes.counted h) law obj program) 1 := by
  apply countC.mono
  · unfold HmRow.templateStrataWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply descParse_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply descParse_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem stratifiedLaw_cost (obj : Json) (program : ByteArray) (k : Nat) :
    countC.P (HmRow.stratifiedLawWith (Hashes.counted h) obj program k) 1 := by
  apply countC.mono
  · unfold HmRow.stratifiedLawWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply templateStrata_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply templateStrata_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem workLaw_cost (obj : Json) (program : ByteArray) (k : Nat) (table : Array (String × Nat × Nat)) :
    countC.P (HmRow.workLawWith (Hashes.counted h) obj program k table) 1 := by
  apply countC.mono
  · unfold HmRow.workLawWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply templateStrata_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply templateStrata_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem deriveTemplate_cost (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    countC.P (HmRow.deriveTemplateUnitsWith (Hashes.counted h) c obj program indices) 1 := by
  apply countC.mono
  · unfold HmRow.deriveTemplateUnitsWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply descParse_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply descParse_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

theorem deriveQword_cost (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    countC.P (HmRow.deriveQwordUnitsWith (Hashes.counted h) c obj program indices) 1 := by
  apply countC.mono
  · unfold HmRow.deriveQwordUnitsWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply decode_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply decode_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem deriveQcall_cost (c : Circuit) (obj : Json) (program : ByteArray) (indices : Array Nat) :
    countC.P (HmRow.deriveQcallUnitsWith (Hashes.counted h) c obj program indices) 1 := by
  apply countC.mono
  · unfold HmRow.deriveQcallUnitsWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply decode_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply decode_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

theorem qwordFinding_cost (c : Circuit) (pub : Public) (obj : Json) (program : ByteArray) (f : HmRow.PartitionFinding) :
    countC.P (HmRow.qwordFindingWith (Hashes.counted h) c pub obj program f) 1 := by
  apply countC.mono
  · unfold HmRow.qwordFindingWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply deriveQword_cost h)
      | (with_reducible apply deriveQcall_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply deriveQword_cost h)
        | (with_reducible apply deriveQcall_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

set_option maxHeartbeats 4000000 in
theorem linksCheck_cost (c : Circuit) (pop : Public) (program : ByteArray) :
    countC.P (HmLinks.checkWith (Hashes.counted h) c pop program) 1 := by
  apply countC.mono
  · unfold HmLinks.checkWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply decode_cost h)
      | (with_reducible apply throw0)
      | (with_reducible apply forIn_range0)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply decode_cost h)
        | (with_reducible apply throw0)
        | (with_reducible apply forIn_range0)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

set_option maxHeartbeats 4000000 in
theorem loadPublic_cost (tags : Tags) (c : Circuit) (file : ByteArray) :
    countC.P (HmRow.loadPublicWith (Hashes.counted h) tags c file) 1 := by
  apply countC.mono
  · unfold HmRow.loadPublicWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply publicDigest_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply publicDigest_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

theorem inLoadPublic_cost (tags : Tags) (c : Circuit) (pi : HmIn.Ports) (file : ByteArray) :
    countC.P (HmIn.loadPublicWith (Hashes.counted h) tags c pi file) 1 := by
  apply countC.mono
  · unfold HmIn.loadPublicWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply loadPublic_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply loadPublic_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

set_option maxHeartbeats 4000000 in
theorem drawn_cost (c : Circuit) (pub : Public) (draw : Json) :
    countC.P (HmRow.drawnWith (Hashes.counted h) c pub draw) 1 := by
  obtain ⟨header, n, digests, outputs, sha, population, tables⟩ := pub
  unfold HmRow.drawnWith
  cases tables
  all_goals
    dsimp only
    apply countC.mono
    · repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply publicDigest_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
      iterate 40
        all_goals try (dsimp +zetaDelta -zeta only)
        all_goals repeat (first
          | extract_lets
          | (with_reducible apply throwBind)
          | (with_reducible apply Cost.bind)
          | (with_reducible apply bindAny)
          | (with_reducible apply publicDigest_cost h)
          | (with_reducible intro _)
          | (with_reducible apply Cost.pure)
          | (with_reducible apply pureAny)
          | (with_reducible apply throw0)
          | (with_reducible apply Cost.throw)
          | (with_reducible apply Cost.lift)
          | (with_reducible apply liftAny)
          | (with_reducible apply Cost.forIn_array)
          | (with_reducible apply Cost.forIn_array0)
          | (with_reducible apply forIn_array0Any)
          | (with_reducible apply Cost.forIn_list)
          | (with_reducible apply Cost.forIn_list0)
          | (with_reducible apply forIn_list0Any)
          | (with_reducible apply forIn_range0)
          | (with_reducible apply forIn_range0Any)
          | (with_reducible apply Cost.mapM_array)
          | (with_reducible apply mapM_array0Any)
          | (with_reducible apply Cost.ite)
          | (with_reducible apply iteAny)
          | (with_reducible apply diteMax)
          | (with_reducible apply diteAny)
          | split)
      all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
    · (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero]); omega

theorem rowParse_cost (tags : Tags) (bytes : ByteArray) (tables : List (String × ByteArray)) :
    countC.P (HmRow.parseWith (Hashes.counted h) tags bytes tables) 2 := by
  apply countC.mono
  · unfold HmRow.parseWith
    repeat (first
      | extract_lets
      | (with_reducible apply throwBind)
      | (with_reducible apply Cost.bind)
      | (with_reducible apply bindAny)
      | (with_reducible apply sha_cost h)
      | (with_reducible apply classOf_cost h)
      | (with_reducible intro _)
      | (with_reducible apply Cost.pure)
      | (with_reducible apply pureAny)
      | (with_reducible apply throw0)
      | (with_reducible apply Cost.throw)
      | (with_reducible apply Cost.lift)
      | (with_reducible apply liftAny)
      | (with_reducible apply Cost.forIn_array)
      | (with_reducible apply Cost.forIn_array0)
      | (with_reducible apply forIn_array0Any)
      | (with_reducible apply Cost.forIn_list)
      | (with_reducible apply Cost.forIn_list0)
      | (with_reducible apply forIn_list0Any)
      | (with_reducible apply forIn_range0)
      | (with_reducible apply forIn_range0Any)
      | (with_reducible apply Cost.mapM_array)
      | (with_reducible apply mapM_array0Any)
      | (with_reducible apply Cost.ite)
      | (with_reducible apply iteAny)
      | (with_reducible apply diteMax)
      | (with_reducible apply diteAny)
      | split)
    iterate 40
      all_goals try (dsimp +zetaDelta -zeta only)
      all_goals repeat (first
        | extract_lets
        | (with_reducible apply throwBind)
        | (with_reducible apply Cost.bind)
        | (with_reducible apply bindAny)
        | (with_reducible apply sha_cost h)
        | (with_reducible apply classOf_cost h)
        | (with_reducible intro _)
        | (with_reducible apply Cost.pure)
        | (with_reducible apply pureAny)
        | (with_reducible apply throw0)
        | (with_reducible apply Cost.throw)
        | (with_reducible apply Cost.lift)
        | (with_reducible apply liftAny)
        | (with_reducible apply Cost.forIn_array)
        | (with_reducible apply Cost.forIn_array0)
        | (with_reducible apply forIn_array0Any)
        | (with_reducible apply Cost.forIn_list)
        | (with_reducible apply Cost.forIn_list0)
        | (with_reducible apply forIn_list0Any)
        | (with_reducible apply forIn_range0)
        | (with_reducible apply forIn_range0Any)
        | (with_reducible apply Cost.mapM_array)
        | (with_reducible apply mapM_array0Any)
        | (with_reducible apply Cost.ite)
        | (with_reducible apply iteAny)
        | (with_reducible apply diteMax)
        | (with_reducible apply diteAny)
        | split)
    all_goals (try simp only [Nat.mul_zero, Nat.zero_mul, Nat.add_zero, Nat.zero_add, Nat.max_self, Nat.zero_max, Nat.max_zero] at *); all_goals omega
  · omega

end RecDeploy.SetupCount
