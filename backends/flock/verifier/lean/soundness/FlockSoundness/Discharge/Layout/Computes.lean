import FlockSoundness.Discharge.Layout.Keyed

set_option Elab.async false

/-!
# `rs.Computes` at the production program's places, from the rows' commit strings

`Binding.HmRows.Computes` (the headline's `hHm`) says: at each row a drawn unit reads, in a satisfying witness of its
table, the `b ‖ c` at the row's `hm96` slot output is the commit string of the row's bytes and the salt the witness
holds. At `keyedRows` (and `keyedRowsOuts`) that is exactly [e2e-hm]'s `port_commit` at each drawn unit's site
(`PortCommits`): the 128 bytes at the `hm96` slot's output (`bcSlotCol`, [e2e-hm]'s `Discharge.Hm.bcCol`) are
`commitString (rowPrefix words ‖ the 2 · words bytes at Typed.msgCol) (the 192 bytes at saltCol)`, with the block's bits
at the site.

* `bcAddr_eq_bcSlotCol`: the address the regions pin (`bcAddr`) is the slot's output column, in a circuit the verifier's
  checks accept (the `hm96` net has `nextPow2 ports` slots per VU);
* `keyedRows_computes`, `keyedRowsOuts_computes`: `Computes` from `PortCommits`.
-/

namespace FlockSoundness.Discharge.Layout

open Flock FlockLevel3 Binding Placed

/-- **`b ‖ c` byte `j`, bit `t`** of port `p`'s row at VU `g`: its `hm96` slot's output bit `8 j + t` ([e2e-hm]'s
`Discharge.Hm.bcCol`, the same term). -/
def bcSlotCol (c : Circuit) (g p j t : ℕ) : ℕ :=
  c.slotCol c.hmNet g p + ((portCols (c.nets.getD c.hmNet default).2.outGroups).getD 0 default).1 + 8 * j + t

/-- The `hm96` net has a power of two of slots per VU, `nextPow2 ports`. -/
theorem hm_perVu {c : Circuit} (C : CheckFacts c) :
    c.perVu.getD c.hmNet 0 = 2 ^ (c.perVu.getD c.hmNet 1).log2 := by
  have h := C.perVu_hm
  unfold nextPow2 at h
  have h1 : c.perVu.getD c.hmNet 1 = c.perVu.getD c.hmNet 0 := by
    by_cases hl : c.hmNet < c.perVu.size
    · simp [Array.getD, hl]
    · simp only [Array.getD, hl, ↓reduceDIte] at h
      exact absurd h (Nat.ne_of_lt (Nat.two_pow_pos _))
  rw [h1, h, Nat.log2_two_pow]

/-- **The address the regions pin is the `hm96` slot's output**, in a circuit the verifier's checks accept. -/
theorem bcAddr_eq_bcSlotCol {c : Circuit} (C : CheckFacts c) (g p j t : ℕ) :
    bcAddr c g p j t = bcSlotCol c g p j t :=
  bcAddr_eq_slotCol c g p j t (hm_perVu C)

variable {c : Flock.Circuit} {hU : Built c.unit} {n : ℕ} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {si : Placed.Sites plan tab}

/-- **Each row's commit string at the drawn units' sites** ([e2e-hm]'s `port_commit` there): in a satisfying witness
of a drawn unit's table, at its block and VU, row port `q`'s 128 bytes at the `hm96` slot's output (`bcSlotCol`) are
the commit string of the row prefix of `words` and the `2 · words` bytes at the message columns (`Typed.msgCol`),
under the 192 bytes at `saltCol`. -/
def PortCommits (c : Flock.Circuit) (si : Placed.Sites plan tab) (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ) :
    Prop :=
  ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M → ∀ q : Fin c.ports.size,
    bytesAt (Binding.wit plan tab S R u M) (fun (j : Fin 128) (t : Fin 8) => siteAddr (si.acc S R u h) (si.o S R u h)
        (bcSlotCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)) =
      commitString ((HmRow.rowPrefix c.ports[q].words).data.toList ++
          bytesAt (Binding.wit plan tab S R u M) (fun (j : Fin (2 * c.ports[q].words)) (t : Fin 8) =>
            siteAddr (si.acc S R u h) (si.o S R u h)
              (Typed.msgCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)))
        (bytesAt (Binding.wit plan tab S R u M) (fun (j : Fin 192) (t : Fin 8) =>
          siteAddr (si.acc S R u h) (si.o S R u h) (saltCol (si.acc S R u h).st.c (si.vu S R u h) q.val j.val t.val)))

variable (hparse : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
    HmRow.parse (si.acc S R u h).tags (si.acc S R u h).cf (si.acc S R u h).tables = .ok c)
  (saltCol : Flock.Circuit → ℕ → ℕ → ℕ → ℕ → ℕ)

variable {outs : Finset (Fin (keyN c hU n))} {popOf : Reg → Public} in
/-- **`Computes` at the production program's places**: `keyedRows`' rows compute their commit strings wherever
[e2e-hm]'s `port_commit` holds at the drawn units' sites (`PortCommits`). -/
theorem keyedRows_computes (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (houts : OutsOut c hU n outs) (hkeys : KeysRow c hU (portRowW c)) (hpc : PortCommits c si saltCol) :
    (keyedRows hparse saltCol ho hsite houts hkeys).Computes := by
  rintro ⟨S, R, u, M, g⟩ ⟨hu, hM, hg, hrow⟩
  dsimp only at hu hM hg hrow ⊢
  obtain ⟨x, rfl⟩ := exists_col_of_io ((keyPlace hparse outs) S R u hu).inst hg
  rcases hr : keyRowOf c hU n (portRowW c) ((keyPlace hparse outs S R u hu).inst.wire x) with _ | ⟨u', q, k⟩
  · have hp : (keyedLayout hparse outs saltCol popOf).pos ((keyPlace hparse outs S R u hu).inst.wire x) =
        .inr ((keyPlace hparse outs S R u hu).inst.wire x) := by
      show (keyReaders (portRowW c) hparse outs saltCol popOf (outBitOf c hU n outs popOf)).pos _ = _
      unfold keyReaders
      exact ofRows_pos_none _ hr
    rw [hp] at hrow
    change Sum.isLeft (Sum.inr _) = true at hrow
    simp at hrow
  obtain ⟨hu', -, -⟩ := keyRowOf_wire outs x hr
  subst u'
  have hp : (keyedLayout hparse outs saltCol popOf).pos ((keyPlace hparse outs S R u hu).inst.wire x) = .inl (u, q) := by
    show (keyReaders (portRowW c) hparse outs saltCol popOf (outBitOf c hU n outs popOf)).pos _ = _
    unfold keyReaders
    exact ofRows_pos_some _ hr
  rw [hp]
  have hcc := (si.acc S R u hu).c_eq (hparse S R u hu)
  have hpar := hparse S R u hu
  rw [← hcc] at hpar
  have P := parse_facts hpar
  obtain ⟨ls, hls⟩ := P.check
  have C := check_facts P.typed hls
  have hbc : bytesAt (Binding.wit plan tab S R u M) ((keyedRows hparse saltCol ho hsite houts hkeys).bcAt S R u (.inl (u, q))) =
      bytesAt (Binding.wit plan tab S R u M) (fun (j : Fin 128) (t : Fin 8) => siteAddr (si.acc S R u hu) (si.o S R u hu)
        (bcSlotCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val)) := by
    show bytesAt _ (keyBcAddr si S R u q) = _
    unfold keyBcAddr
    simp only [hu, ↓reduceDIte, bcAddr_eq_bcSlotCol C]
  have hrowE := keyRow_enc (rowW := portRowW c) (hparse := hparse) (outs := outs) (saltCol := saltCol) (popOf := popOf)
    (outBit := outBitOf c hU n outs popOf) bitCode hu M u q (words := c.ports[q].words) rfl
  have hsalt : (keyedLayout hparse outs saltCol popOf).salt S R u M (.inl (u, q)) =
      bytesAt (Binding.wit plan tab S R u M) (fun (j : Fin 192) (t : Fin 8) =>
        siteAddr (si.acc S R u hu) (si.o S R u hu) (saltCol (si.acc S R u hu).st.c (si.vu S R u hu) q.val j.val t.val)) := by
    show bytesAt _ (keySaltAddr (hU := hU) si saltCol S R u (.inl (u, q))) = _
    unfold keySaltAddr
    simp only [hu, ↓reduceDIte]
  have e := hpc S R u hu M hM q
  unfold commitString at e
  show bytesAt (Binding.wit plan tab S R u M) ((keyedRows hparse saltCol ho hsite houts hkeys).bcAt S R u (.inl (u, q))) =
    Merkle.xorBytes (digBytes (Refine.H512 (rowEnc ((keyedLayout hparse outs saltCol popOf).row S R u M (.inl (u, q))))))
      (Refine.K512.mask ((keyedLayout hparse outs saltCol popOf).salt S R u M (.inl (u, q)))) ++
    digBytes (Refine.H512 (Refine.K512.saltPrefix ++ (keyedLayout hparse outs saltCol popOf).salt S R u M (.inl (u, q))))
  rw [hbc, hrowE, hsalt]
  exact e

variable {popOf : Reg → Public} in
/-- **`Computes` at the production program's places, with the returned outputs' bits for the program's outputs**. -/
theorem keyedRowsOuts_computes {tags : Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    (hc : HmRow.parse tags bytes tables = .ok c) (h1 : c.unit.inGroups.size = 1) (hpr : PortReads c hU)
    (ho : c.outNet = c.unitNet)
    (hsite : ∀ S R u (h : u ∈ S) M, Binding.Sat plan tab S R u M →
      BcSite (si.acc S R u h) (popOf R) (si.o S R u h) (si.vu S R u h) u.val)
    (hpc : PortCommits c si saltCol) :
    (keyedRowsOuts hparse saltCol hc h1 hpr ho hsite).Computes :=
  keyedRows_computes hparse saltCol ho hsite (outsOut_outsOf c hU n) (keysRow_of_parse hc h1 hpr) hpc

end FlockSoundness.Discharge.Layout
