import Properties.Composition
import Proofs.Properties.Pouw.LiveSalt.Canon
import Proofs.Flock.Claims.Complete.Json.Hex

/-!
# The composition's properties, proved

`answerAccepts`, `acceptMeans`, `acceptSame` and `acceptTied` of `Properties.Composition`, by unfolding
`Core.Protocol.answer`, `Core.Composition.check` into `run` and `judge`, `Core.Composition.checkSame` into `strays` and
then `check`, and `Core.Composition.holdTies` into its ties' `strays`; a `strays` that finds nothing is one value for all
(`oneRecord_of_strays`, `tied_of_strays`). `bindCollision`, because `bindMessage` is injective in the digests: past the id
and the zero byte, its JSON reads back entry by entry, each escaped string ending at its first unescaped quote
(`canonStr_split`) and lowercase hex decoding to its bytes (`unhex_hexOf`).

`acceptSame_honest` and `acceptTied_honest` are their honest lemmas, an input the compiled function accepts, and
`acceptSame_nontrivial` and `acceptTied_nontrivial` their nontrivial ones, an input whose conclusion fails, which a
verifier that accepts everything accepts.
-/

open private hex4 from Flock.Canon

namespace Properties.Composition

open Core Core.Composition

/-- `answer` accepts exactly when the three decode and the check accepts what they decode to. -/
theorem answer_accepts_iff (P : Protocol) (a b e : ByteArray) :
    (P.answer a b e).verdict = .accept ↔ DecodedAccepted P a b e := by
  unfold Protocol.answer
  constructor
  · intro h
    cases hp : (canonicalJson a >>= P.params) with
    | error _ => rw [hp] at h; cases h
    | ok p =>
      rw [hp] at h
      cases hc : P.commitment b with
      | error _ => rw [hc] at h; cases h
      | ok c =>
        rw [hc] at h
        cases he : P.evidence e with
        | error _ => simp only [he] at h; cases h
        | ok ev =>
          simp only [he] at h
          cases hch : P.check p c ev with
          | accept pub prof => exact ⟨p, c, ev, hp, hc, he, pub, prof, hch⟩
          | reject _ => rw [hch] at h; cases h
          | undecided _ => rw [hch] at h; cases h
  · rintro ⟨p, c, ev, hp, hc, he, pub, prof, hch⟩
    simp only [hp, hc, he, hch]

theorem answerAccepts : AnswerAccepts := fun P a b e => (answer_accepts_iff P a b e).mp

/-- An accepted answer's digest is that of the commitment as the protocol decodes it. -/
theorem answer_digest {P : Protocol} {a b e : ByteArray} (h : (P.answer a b e).verdict = .accept) :
    (P.answer a b e).digest = digestOf P b := by
  obtain ⟨p, c, ev, hp, hc, he, pub, prof, hch⟩ := answerAccepts P a b e h
  simp only [Protocol.answer, digestOf, hp, hc, he, hch, Except.toOption, Option.map]

theorem run_ok {cs : List Protocol} {ps : List Named} {ev : List Case} {answers : Answers}
    (h : run cs ps ev = .ok answers) :
    ps.map (·.id) = cs.map (·.id) ∧ ev.map (·.id) = cs.map (·.id) ∧
      (∀ x ∈ triples cs ps ev, x.2.2.params = x.2.1.params) ∧ answers = answersOf cs ps ev := by
  unfold run at h
  by_cases h1 : ps.map (·.id) = cs.map (·.id)
  · by_cases h2 : ev.map (·.id) = cs.map (·.id)
    · simp only [h1, h2, ne_eq, not_true_eq_false, ite_false] at h
      split at h
      · cases h
      · rename_i hf
        refine ⟨h1, h2, ?_, ?_⟩
        · intro x hx
          have hx' := List.find?_eq_none.mp hf x hx
          obtain ⟨P, n, k⟩ := x
          simpa using hx'
        · cases h; rfl
    · simp only [h1, h2, ne_eq, not_true_eq_false, not_false_eq_true, ite_false, ite_true] at h
      cases h
  · simp only [h1, ne_eq, not_false_eq_true, ite_true] at h
    cases h

theorem judge_accept {id : String} {c : ByteArray} {answers : Answers} {pub prof : Keyed}
    (h : judge id c answers = .accept pub prof) :
    (∀ x ∈ answers, x.2.verdict = .accept) ∧ binding id answers = some c ∧
      pub = answers.map (fun (i, a) => (i, a.published.getD .null)) ∧
      prof = answers.map (fun (i, a) => (i, a.profile.getD .null)) := by
  unfold judge at h
  split at h
  · cases h
  rename_i hr
  split at h
  · cases h
  rename_i hb
  split at h
  · cases h
  rename_i hu
  cases h
  refine ⟨?_, Decidable.of_not_not hb, rfl, rfl⟩
  intro x hx
  simp only [List.any_eq_true, not_exists, not_and, decide_eq_true_eq] at hr hu
  cases hv : x.2.verdict with
  | accept => rfl
  | reject => exact (hr x hx hv).elim
  | undecided => exact (hu x hx hv).elim

theorem mapM_map_congr.{u, v} {α : Type u} {β : Type v} {γ : Type} (l : List α) (f : α → β) (g : β → Option γ) (k : α → Option γ)
    (hfg : ∀ x ∈ l, g (f x) = k x) : (l.map f).mapM g = l.mapM k := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    simp only [List.map_cons, List.mapM_cons]
    rw [hfg a (by simp), ih (fun x hx => hfg x (by simp [hx]))]

theorem acceptMeans : AcceptMeans := by
  intro id cs ps c ev pub prof h
  unfold check at h
  split at h
  · cases h
  rename_i answers hrun
  obtain ⟨h1, h2, hpar, rfl⟩ := run_ok hrun
  obtain ⟨hacc, hbind, rfl, rfl⟩ := judge_accept h
  have hlen1 : ps.length = cs.length := by simpa using congrArg List.length h1
  have hlen2 : ev.length = cs.length := by simpa using congrArg List.length h2
  have hfst : (triples cs ps ev).map (·.1) = cs :=
    List.map_fst_zip (by simp [List.length_zip, hlen1, hlen2])
  have hone : ∀ x ∈ triples cs ps ev,
      (x.1.answer x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence).verdict = .accept := by
    intro x hx
    have := hacc _ (List.mem_map_of_mem (f := fun (x : Protocol × Named × Case) =>
      (x.1.id, x.1.answer x.2.1.params.toUTF8 x.2.2.commitment x.2.2.evidence)) hx)
    exact this
  have hkeys : ∀ g : String × Answer → String × Lean.Json, (∀ x, (g x).1 = x.1) →
      ((answersOf cs ps ev).map g).map (·.1) = cs.map (·.id) := by
    intro g hg
    rw [List.map_map]
    have hm : (answersOf cs ps ev).map ((·.1) ∘ g) = (triples cs ps ev).map (·.1.id) := by
      unfold answersOf triples
      rw [List.map_map]
      exact List.map_congr_left (fun x _ => by simp only [Function.comp_apply, hg]; try rfl)
    rw [hm]
    exact (List.map_map ..).symm.trans (congrArg (List.map (·.id)) hfst)
  refine ⟨⟨h1, h2, hfst, hpar⟩, fun x hx => answerAccepts _ _ _ _ (hone x hx), ?_,
    hkeys _ (fun _ => rfl), hkeys _ (fun _ => rfl)⟩
  unfold binding at hbind
  unfold digests
  unfold answersOf at hbind
  rw [← hbind]
  congr 1
  refine (mapM_map_congr _ _ _ _ ?_).symm
  intro x hx
  have hd := answer_digest (hone x hx)
  obtain ⟨P, n, k⟩ := x
  simp only at hd ⊢
  rw [hd]

theorem mapM_none {α β : Type} {f : α → Option β} : ∀ {l : List α}, l.mapM f = none → ∃ x ∈ l, f x = none
  | [], h => by simp at h
  | a :: l, h => by
    cases ha : f a with
    | none => exact ⟨a, List.mem_cons.2 (Or.inl rfl), ha⟩
    | some b =>
      cases hl : l.mapM f with
      | none =>
        obtain ⟨x, hx, e⟩ := mapM_none hl
        exact ⟨x, List.mem_cons.2 (Or.inr hx), e⟩
      | some zs => simp [List.mapM_cons, ha, hl] at h

theorem mapM_some {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {ys : List β}, l.mapM f = some ys → ∀ x ∈ l, ∃ y ∈ ys, f x = some y
  | [], _, _, x, hx => absurd hx List.not_mem_nil
  | a :: l, ys, h, x, hx => by
    cases ha : f a with
    | none => simp [List.mapM_cons, ha] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [List.mapM_cons, ha, hl] at h
      | some zs =>
        simp only [List.mapM_cons, ha, hl, Option.pure_def, Option.bind_eq_bind, Option.bind_some,
          Option.some.injEq] at h
        subst h
        rcases List.mem_cons.1 hx with rfl | hx
        · exact ⟨b, List.mem_cons.2 (Or.inl rfl), ha⟩
        · obtain ⟨y, hy, e⟩ := mapM_some hl x hx
          exact ⟨y, List.mem_cons.2 (Or.inr hy), e⟩

theorem eq_of_bne_false {a b : ByteArray} (h : (a != b) = false) : a = b := by
  have h' : (a.data == b.data) = true := by simpa [bne, BEq.beq, ByteArray.beq] using h
  exact ByteArray.ext (by simpa using h')

/-- A `mapM` whose entries past the first all have the first's second component: either `f` has no value on some
entry, or every entry's value has that one. -/
theorem one_of_mapM {α : Type} {f : α → Option (String × ByteArray)} {l : List α}
    (h : ∀ s d rest, l.mapM f = some ((s, d) :: rest) → rest.find? (fun x => x.2 != d) = none) :
    (∃ x ∈ l, f x = none) ∨ ∃ d, ∀ x ∈ l, ∃ s, f x = some (s, d) := by
  cases hm : l.mapM f with
  | none => exact Or.inl (mapM_none hm)
  | some ys =>
    match ys, hm with
    | [], hm =>
      refine Or.inr ⟨ByteArray.empty, fun x hx => ?_⟩
      obtain ⟨y, hy, -⟩ := mapM_some hm x hx
      exact absurd hy List.not_mem_nil
    | (s, d) :: rest, hm =>
      refine Or.inr ⟨d, fun x hx => ?_⟩
      obtain ⟨⟨t, d'⟩, hy, e⟩ := mapM_some hm x hx
      refine ⟨t, ?_⟩
      rw [e]
      rcases List.mem_cons.1 hy with hy | hy
      · cases hy; rfl
      · have hf : (d' != d) = false := by simpa using List.find?_eq_none.1 (h s d rest hm) _ hy
        rw [eq_of_bne_false hf]

theorem oneRecord_of_strays {ss : List String} {cs : List Protocol} {ev : List Case}
    (h : strays ss cs ev = none) : OneRecord ss cs ev := by
  have h' := one_of_mapM (f := fun s => (committedBy s cs ev).map (s, ·)) (l := ss) (fun s d rest hm => by
    unfold strays at h
    rw [hm] at h
    simpa using h)
  rcases h' with ⟨s, hs, e⟩ | ⟨d, hd⟩
  · exact Or.inl ⟨s, hs, by simpa using e⟩
  · refine Or.inr ⟨d, fun s hs => ?_⟩
    obtain ⟨t, e⟩ := hd s hs
    cases hc : committedBy s cs ev with
    | none => rw [hc] at e; cases e
    | some d' => rw [hc] at e; cases e; rfl

theorem tied_of_strays {t : Tie} {ev : List Case} (h : t.strays ev = none) : Tied t ev := by
  have h' := one_of_mapM (f := fun n => (named ev n).map (n.1, ·)) (l := t.names) (fun s d rest hm => by
    have hm' : t.names.mapM (fun n : String × (ByteArray → Option ByteArray) =>
        ((ev.find? fun (k : Case) => k.id == n.1).bind fun k => n.2 k.commitment).map (n.1, ·)) =
        some ((s, d) :: rest) := hm
    unfold Tie.strays at h
    simp only [hm'] at h
    simpa using h)
  rcases h' with ⟨n, hn, e⟩ | ⟨d, hd⟩
  · exact Or.inl ⟨n, hn, by simpa using e⟩
  · refine Or.inr ⟨d, fun n hn => ?_⟩
    obtain ⟨s, e⟩ := hd n hn
    cases hc : named ev n with
    | none => rw [hc] at e; cases e
    | some d' => rw [hc] at e; cases e; rfl

theorem acceptSame : AcceptSame := by
  intro id ss cs ps c ev pub prof h
  unfold checkSame at h
  split at h
  · cases h
  rename_i answers hrun
  split at h
  · cases h
  rename_i hs
  have hc : check id cs ps c ev = .accept pub prof := by
    unfold check
    rw [hrun]
    exact h
  exact ⟨acceptMeans _ _ _ _ _ _ _ hc, oneRecord_of_strays hs⟩

theorem findSome_none {t : List Tie} {ev : List Case} (h : t.findSome? (·.strays ev) = none) :
    ∀ x ∈ t, x.strays ev = none := by
  induction t with
  | nil => intro x hx; exact absurd hx List.not_mem_nil
  | cons a t ih =>
    intro x hx
    rw [List.findSome?_cons] at h
    cases ha : a.strays ev with
    | some _ => rw [ha] at h; cases h
    | none =>
      rw [ha] at h
      rcases List.mem_cons.1 hx with rfl | hx
      · exact ha
      · exact ih h x hx

theorem acceptTied : AcceptTied := by
  intro ties ev o pub prof h
  cases o with
  | reject r => cases h
  | accept p q =>
    simp only [holdTies] at h
    split at h
    · cases h
    rename_i hf
    exact ⟨h, fun t ht => tied_of_strays (findSome_none hf t ht)⟩
  | undecided r =>
    simp only [holdTies] at h
    split at h
    · cases h
    · cases h

/-- Ties over no components hold every outcome as it is. -/
theorem holdTies_nil (ev : List Case) (o : Outcome Keyed Keyed) : holdTies [] ev o = o := by
  cases o <;> rfl

/-- **`AcceptSame`'s honest lemma**: remote's `checkSame` accepts a composition that fits, with no components and no
record subprotocols, under the binding of no digests. -/
theorem acceptSame_honest :
    checkSame "verity/remote/v0" [] [] [] (bind "verity/remote/v0" []) [] = .accept [] [] := by
  simp [checkSame, run, strays, judge, binding, answersOf]

/-- **`AcceptSame`'s nontrivial lemma**: `SameAccepted` fails at parameters that name a component the composition
doesn't have, which a verifier that accepts everything accepts. -/
theorem acceptSame_nontrivial :
    ¬ SameAccepted "verity/remote/v0" [] [] [⟨"x", ""⟩] ByteArray.empty [] [] [] := by
  rintro ⟨⟨⟨h, -⟩, -⟩, -⟩
  simp at h

/-- **`AcceptTied`'s honest lemma**: `holdTies` accepts an accepted outcome whose ties hold. -/
theorem acceptTied_honest : holdTies [] [] (.accept [] []) = .accept [] [] :=
  holdTies_nil _ _

/-- **`AcceptTied`'s nontrivial lemma**: `TiedAccepted` fails at an outcome that rejected, which a verifier that accepts
everything accepts. -/
theorem acceptTied_nontrivial : ¬ TiedAccepted [] [] (.reject "") [] [] := by
  rintro ⟨h, -⟩
  cases h

/-! ## `Flock.canonStr`'s strings, read back

`canonStr` writes each character as itself, a short escape, or a `\u` escape (two, a UTF-16 surrogate pair, above
`0xFFFF`): `unesc` reads one back (`unesc_esc`), so an escaped string ends at its first unescaped quote
(`canonStr_split`). -/

namespace Esc

open Properties.Pouw.LiveSalt.Canon (unhexL nib)

/-- What `Flock.canonStr` writes for one character. -/
def esc (c : Char) : List Char :=
  match c with
  | '"' => ['\\', '"']
  | '\\' => ['\\', '\\']
  | '\n' => ['\\', 'n']
  | '\r' => ['\\', 'r']
  | '\t' => ['\\', 't']
  | c => if c.toNat == 8 then ['\\', 'b'] else if c.toNat == 12 then ['\\', 'f']
    else if c.toNat < 0x20 || 0x7f ≤ c.toNat then (Flock.escU c.toNat).toList else [c]

theorem foldl_esc (f : String → Char → String) (hf : ∀ b a, (f b a).toList = b.toList ++ esc a) :
    ∀ (cs : List Char) (out : String), (cs.foldl f out).toList = out.toList ++ cs.flatMap esc := by
  intro cs
  induction cs with
  | nil => intro out; simp
  | cons c cs ih => intro out; rw [List.foldl_cons, ih, hf, List.flatMap_cons, List.append_assoc]

theorem canonStr_toList (s : String) : (Flock.canonStr s).toList = '"' :: s.toList.flatMap esc ++ ['"'] := by
  unfold Flock.canonStr
  simp only [Id.run, List.forIn_pure_yield_eq_foldl]
  show (List.foldl _ "\"" s.toList ++ "\"").toList = _
  rw [String.toList_append, foldl_esc]
  · simp
  · intro b a
    unfold esc
    split
    all_goals first
      | (simp; done)
      | ((repeat' split) <;> simp_all)

/-- `hex4`'s digit of `x`. -/
def dg (x : Nat) : Char := if x < 10 then Char.ofNat (48 + x) else Char.ofNat (87 + x)

theorem nib_dg : ∀ x, x < 16 → nib (dg x) = x := by decide

theorem hex4_toList (n : Nat) :
    (hex4 n).toList = [dg ((n >>> 12) % 16), dg ((n >>> 8) % 16), dg ((n >>> 4) % 16), dg ((n >>> 0) % 16)] := by
  simp only [hex4, String.toList_ofList, dg] <;> rfl

/-- `hex4` of a number below `0x10000` is four hex digits that read back to it. -/
theorem hex4_ok (n : Nat) (hn : n < 0x10000) : (hex4 n).toList.length = 4 ∧ unhexL (hex4 n).toList = n := by
  rw [hex4_toList]
  refine ⟨rfl, ?_⟩
  have e12 : n >>> 12 = n / 4096 := by rw [Nat.shiftRight_eq_div_pow]; rfl
  have e8 : n >>> 8 = n / 256 := by rw [Nat.shiftRight_eq_div_pow]; rfl
  have e4 : n >>> 4 = n / 16 := by rw [Nat.shiftRight_eq_div_pow]; rfl
  have e0 : n >>> 0 = n := Nat.shiftRight_zero
  simp only [unhexL, List.foldl_cons, List.foldl_nil, nib_dg _ (Nat.mod_lt _ (by decide : 16 > 0)), e12, e8, e4, e0]
  omega

/-- Whether a `\u` escape's number is a high surrogate, the first of a pair. -/
def high (n : Nat) : Bool := 0xD800 ≤ n && n < 0xDC00

/-- The character a surrogate pair `h`, `l` stands for. -/
def pair (h l : Nat) : Nat := 0x10000 + (h - 0xD800) * 0x400 + (l - 0xDC00)

/-- Reads a surrogate pair's second `\u` escape. -/
def unescLow : List Char → Option (Nat × List Char)
  | '\\' :: 'u' :: a :: b :: c :: d :: r => some (unhexL [a, b, c, d], r)
  | _ => none

/-- Reads one escape's character after its backslash. -/
def unesc2 : List Char → Option (Char × List Char)
  | '"' :: r => some ('"', r)
  | '\\' :: r => some ('\\', r)
  | 'n' :: r => some ('\n', r)
  | 'r' :: r => some ('\r', r)
  | 't' :: r => some ('\t', r)
  | 'b' :: r => some (Char.ofNat 8, r)
  | 'f' :: r => some (Char.ofNat 12, r)
  | 'u' :: a :: b :: c :: d :: r =>
    if high (unhexL [a, b, c, d]) then (unescLow r).map fun (l, r') => (Char.ofNat (pair (unhexL [a, b, c, d]) l), r')
    else some (Char.ofNat (unhexL [a, b, c, d]), r)
  | _ => none

/-- Reads one character of an escaped string. -/
def unesc : List Char → Option (Char × List Char)
  | [] => none
  | c :: r => if c = '\\' then unesc2 r else some (c, r)

theorem u_toList : ("\\u" : String).toList = ['\\', 'u'] := by decide

theorem unesc_escU (c : Char) (r : List Char) : unesc ((Flock.escU c.toNat).toList ++ r) = some (c, r) := by
  have hv : c.toNat < 0xd800 ∨ (0xdfff < c.toNat ∧ c.toNat < 0x110000) := c.valid
  unfold Flock.escU
  by_cases hlt : c.toNat < 0x10000
  · rw [if_pos hlt]
    have hc : high c.toNat = false := by
      unfold high
      rcases hv with h | ⟨h, -⟩
      · simp [show ¬ 0xD800 ≤ c.toNat by omega]
      · simp [show ¬ c.toNat < 0xDC00 by omega]
    obtain ⟨hl, hx⟩ := hex4_ok _ hlt
    rw [String.toList_append]
    generalize (hex4 c.toNat).toList = xs at hl hx ⊢
    match xs, hl with
    | [a1, a2, a3, a4], _ => simp [u_toList, unesc, unesc2, hx, hc, Char.ofNat_toNat]
  · rw [if_neg hlt]
    have hH : 0xD800 + (c.toNat - 0x10000) / 0x400 < 0x10000 := by omega
    have hL : 0xDC00 + (c.toNat - 0x10000) % 0x400 < 0x10000 := by omega
    have hp : pair (0xD800 + (c.toNat - 0x10000) / 0x400) (0xDC00 + (c.toNat - 0x10000) % 0x400) = c.toNat := by
      unfold pair; omega
    have hc : high (0xD800 + (c.toNat - 0x10000) / 0x400) = true := by
      unfold high
      simp [show 0xD800 ≤ 0xD800 + (c.toNat - 0x10000) / 0x400 by omega,
        show 0xD800 + (c.toNat - 0x10000) / 0x400 < 0xDC00 by omega]
    obtain ⟨hl, hx⟩ := hex4_ok _ hH
    obtain ⟨hl', hy⟩ := hex4_ok _ hL
    simp only [String.toList_append, List.append_assoc]
    generalize 0xD800 + (c.toNat - 0x10000) / 0x400 = H at hl hx hc hp ⊢
    generalize 0xDC00 + (c.toNat - 0x10000) % 0x400 = L at hl' hy hp ⊢
    generalize (hex4 H).toList = xs at hl hx ⊢
    generalize (hex4 L).toList = ys at hl' hy ⊢
    match xs, hl, ys, hl' with
    | [a1, a2, a3, a4], _, [b1, b2, b3, b4], _ =>
      simp [u_toList, unesc, unesc2, unescLow, hx, hy, hc, hp, Char.ofNat_toNat]

theorem unesc_esc (c : Char) (r : List Char) : unesc (esc c ++ r) = some (c, r) := by
  unfold esc
  split
  all_goals first
    | (simp [unesc, unesc2]; done)
    | skip
  rename_i h1 h2 h3 h4 h5
  by_cases h8 : c.toNat = 8
  · obtain rfl : c = Char.ofNat 8 := by rw [← h8, Char.ofNat_toNat]
    simp (config := {decide := true}) [unesc, unesc2]
  by_cases h12 : c.toNat = 12
  · obtain rfl : c = Char.ofNat 12 := by rw [← h12, Char.ofNat_toNat]
    simp (config := {decide := true}) [unesc, unesc2]
  by_cases hu : c.toNat < 0x20 ∨ 0x7f ≤ c.toNat
  · have hb : (decide (c.toNat < 0x20) || decide (0x7f ≤ c.toNat)) = true := by simpa using hu
    simp only [beq_iff_eq, h8, h12, hb, ↓reduceIte]
    exact unesc_escU c r
  · have hb : (decide (c.toNat < 0x20) || decide (0x7f ≤ c.toNat)) = false := by simpa using hu
    simp only [beq_iff_eq, h8, h12, hb, ↓reduceIte, Bool.false_eq_true, List.singleton_append, unesc]
    exact ite_eq_right h2

theorem escs_inj : ∀ (cs cs' r r' : List Char), cs.flatMap esc ++ '"' :: r = cs'.flatMap esc ++ '"' :: r' →
    cs = cs' ∧ r = r' := by
  intro cs
  induction cs with
  | nil =>
    intro cs' r r' h
    cases cs' with
    | nil => simpa using h
    | cons c' cs' =>
      exfalso
      simp only [List.flatMap_nil, List.nil_append, List.flatMap_cons, List.append_assoc] at h
      have hu := congrArg unesc h
      rw [unesc_esc] at hu
      simp only [unesc] at hu
      obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hu)
      simp [esc] at h
  | cons c cs ih =>
    intro cs' r r' h
    cases cs' with
    | nil =>
      exfalso
      simp only [List.flatMap_nil, List.nil_append, List.flatMap_cons, List.append_assoc] at h
      have hu := congrArg unesc h
      rw [unesc_esc] at hu
      simp only [unesc] at hu
      obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj hu)
      simp [esc] at h
    | cons c' cs' =>
      simp only [List.flatMap_cons, List.append_assoc] at h
      have hu := congrArg unesc h
      rw [unesc_esc, unesc_esc] at hu
      obtain ⟨rfl, h2⟩ := Prod.mk.inj (Option.some.inj hu)
      obtain ⟨rfl, rfl⟩ := ih cs' r r' h2
      exact ⟨rfl, rfl⟩

/-- An escaped string ends where it ends: what follows it doesn't change which string it is. -/
theorem canonStr_split {s s' : String} {r r' : List Char}
    (h : (Flock.canonStr s).toList ++ r = (Flock.canonStr s').toList ++ r') : s = s' ∧ r = r' := by
  simp only [canonStr_toList, List.cons_append, List.append_assoc, List.cons.injEq, true_and] at h
  obtain ⟨h1, h2⟩ := escs_inj _ _ _ _ h
  exact ⟨String.toList_inj.mp h1, h2⟩

end Esc

/-! ## `bind`'s messages, read back

Over any quoting `q` that ends where it ends and any injective encoding `h` of the digests: `bodyOf` is the JSON `bind`
writes, and it determines the digests (`body_inj`). -/

namespace BindInj

variable (q : String → String) (h : ByteArray → String)

def entryOf (x : String × ByteArray) : String :=
  "{\"commitment\":" ++ q (h x.2) ++ ",\"id\":" ++ q x.1 ++ "}"

def restOf : List (String × ByteArray) → List Char
  | [] => [']', '}']
  | x :: xs => ',' :: ((entryOf q h x).toList ++ restOf xs)

def itemsOf : List (String × ByteArray) → List Char
  | [] => [']', '}']
  | x :: xs => (entryOf q h x).toList ++ restOf q h xs

theorem intercalate_items (ds : List (String × ByteArray)) :
    (",".intercalate (ds.map (entryOf q h))).toList ++ [']', '}'] = itemsOf q h ds := by
  match ds with
  | [] => simp [itemsOf]
  | [x] => simp [itemsOf, restOf]
  | x :: y :: ds =>
    rw [List.map_cons, List.map_cons, String.intercalate_cons_cons, String.toList_append, String.toList_append]
    have ih := intercalate_items (y :: ds)
    rw [List.map_cons] at ih
    rw [List.append_assoc, List.append_assoc, ih]
    simp [itemsOf, restOf]

theorem entry_split (hq : ∀ s s' r r', (q s).toList ++ r = (q s').toList ++ r' → s = s' ∧ r = r')
    (hh : ∀ a b, h a = h b → a = b) {x y : String × ByteArray} {r r' : List Char}
    (e : (entryOf q h x).toList ++ r = (entryOf q h y).toList ++ r') : x = y ∧ r = r' := by
  simp only [entryOf, String.toList_append, List.append_assoc] at e
  have e1 := List.append_cancel_left e
  obtain ⟨h2, e2⟩ := hq _ _ _ _ e1
  have e3 := List.append_cancel_left e2
  obtain ⟨h1, e4⟩ := hq _ _ _ _ e3
  have e5 := List.append_cancel_left e4
  obtain ⟨x1, x2⟩ := x
  obtain ⟨y1, y2⟩ := y
  exact ⟨Prod.ext h1 (hh _ _ h2), e5⟩

theorem entry_head (x : String × ByteArray) (r : List Char) : ∃ t, (entryOf q h x).toList ++ r = '{' :: t := by
  simp only [entryOf, String.toList_append, List.append_assoc]
  exact ⟨_, rfl⟩

theorem rest_inj (hq : ∀ s s' r r', (q s).toList ++ r = (q s').toList ++ r' → s = s' ∧ r = r')
    (hh : ∀ a b, h a = h b → a = b) :
    ∀ ds ds' : List (String × ByteArray), restOf q h ds = restOf q h ds' → ds = ds'
  | [], [], _ => rfl
  | [], _ :: _, e => by simp [restOf] at e
  | _ :: _, [], e => by simp [restOf] at e
  | x :: xs, y :: ys, e => by
    simp only [restOf, List.cons.injEq, true_and] at e
    obtain ⟨rfl, e'⟩ := entry_split q h hq hh e
    rw [rest_inj hq hh xs ys e']

theorem items_inj (hq : ∀ s s' r r', (q s).toList ++ r = (q s').toList ++ r' → s = s' ∧ r = r')
    (hh : ∀ a b, h a = h b → a = b) :
    ∀ ds ds' : List (String × ByteArray), itemsOf q h ds = itemsOf q h ds' → ds = ds'
  | [], [], _ => rfl
  | [], y :: ys, e => by
    obtain ⟨t, ht⟩ := entry_head q h y (restOf q h ys)
    simp only [itemsOf] at e
    rw [ht] at e
    simp at e
  | x :: xs, [], e => by
    obtain ⟨t, ht⟩ := entry_head q h x (restOf q h xs)
    simp only [itemsOf] at e
    rw [ht] at e
    simp at e
  | x :: xs, y :: ys, e => by
    simp only [itemsOf] at e
    obtain ⟨rfl, e'⟩ := entry_split q h hq hh e
    rw [rest_inj q h hq hh xs ys e']

def bodyOf (ds : List (String × ByteArray)) : String :=
  "{\"components\":[" ++ ",".intercalate (ds.map fun (i, d) =>
    "{\"commitment\":" ++ q (h d) ++ ",\"id\":" ++ q i ++ "}") ++ "]}"

theorem body_inj (hq : ∀ s s' r r', (q s).toList ++ r = (q s').toList ++ r' → s = s' ∧ r = r')
    (hh : ∀ a b, h a = h b → a = b) {ds ds' : List (String × ByteArray)} (e : bodyOf q h ds = bodyOf q h ds') :
    ds = ds' := by
  have e1 := congrArg String.toList e
  have hm : ∀ ds : List (String × ByteArray), (ds.map fun (i, d) =>
      "{\"commitment\":" ++ q (h d) ++ ",\"id\":" ++ q i ++ "}") = ds.map (entryOf q h) := fun _ => rfl
  simp only [bodyOf, hm, String.toList_append, List.append_assoc] at e1
  have e2 := List.append_cancel_left e1
  have hr : ("]}" : String).toList = [']', '}'] := by simp
  rw [hr, intercalate_items, intercalate_items] at e2
  exact items_inj q h hq hh ds ds' e2

theorem msg_inj (id : String) {B B' : String}
    (e : id.toUTF8 ++ ByteArray.mk #[0] ++ B.toUTF8 = id.toUTF8 ++ ByteArray.mk #[0] ++ B'.toUTF8) : B = B' := by
  have e1 := congrArg (fun b : ByteArray => b.data.toList) e
  simp only [ByteArray.data_append, Array.toList_append, List.append_assoc] at e1
  have e2 := List.append_cancel_left (List.append_cancel_left e1)
  exact String.toByteArray_inj.1 (ByteArray.ext (Array.ext' e2))

end BindInj

theorem bindCollision : BindCollision := by
  intro id ds ds'
  by_cases e : bind id ds = bind id ds'
  · refine Or.inr ?_
    by_cases hd : ds = ds'
    · exact Or.inl hd
    · refine Or.inr ⟨fun hm => hd ?_, e⟩
      unfold bindMessage at hm
      refine BindInj.body_inj Flock.canonStr Flock.hexOf
        (fun _ _ _ _ h => Esc.canonStr_split h) (fun a b h => ?_) (BindInj.msg_inj id hm)
      have e := FlockComplete.JsonRT.unhex_hexOf a
      rw [h, FlockComplete.JsonRT.unhex_hexOf] at e
      exact (Except.ok.inj e).symm
  · exact Or.inl e

end Properties.Composition
