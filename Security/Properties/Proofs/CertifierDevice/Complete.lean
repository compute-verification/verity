import Properties.Proofs.CertifierDevice.Late

/-!
# The certifier device, proved: a complete record is the Program's

A complete record's window has only clean ticks up to its close (`clean_of_complete`), so every frame handed in it was
handed at its handing bucket (`window_hands`), and the certifier was unfaulted at each of its ticks, where its `queued`
count is the Program's queue of the handing buckets so far (`hinv`). Each row is then the FIFO's release
(`release`), which is `NetworkCertifier.expectedRow` of the frames handed through the record's publication.
-/

namespace Properties.CertifierDevice.Proofs

open _root_.CertifierDevice
open NetworkCertifier (Reason Status)

theorem completeRows_proof : CompleteRows := by
  intro c tr n k backlog acts rec p hE hacts hmem
  dsimp only
  intro hst hw j hj i hia
  rw [egress_acts hE] at hacts
  obtain rfl := Option.some.inj hacts
  unfold egressActions at hmem
  obtain ⟨hc, rfl⟩ : (k + 1) % c.egress.buckets = 0 ∧
      rec = ⟨c.egressLink, k / c.egress.buckets,
        Array.ofFn fun j : Fin c.egress.buckets => rowOf c tr (k / c.egress.buckets * c.egress.buckets + j),
        statusOf c tr (k / c.egress.buckets)⟩ := by
    rcases List.mem_append.1 hmem with h | h
    · split at h <;> simp at h
    · split at h
      · simp only [List.mem_singleton, Action.publish.injEq] at h
        exact ⟨‹_›, h.1⟩
      · simp at h
  dsimp only at hst hw ⊢
  change k / c.egress.buckets < (pre c tr (n + 1)).shaper.syncs.size at hw
  change NetworkCertifier.expectedRow _ _ (pre c tr (n + 1)).shaper.syncs.size
    (fun w => (pre c tr (n + 1)).shaper.syncs[w]) _ _ _ = _
  obtain ⟨hT0, hkm, hkl⟩ := mod_last hc
  have hn := egressAt_lt hE
  have hgrid : (Array.ofFn fun j : Fin c.egress.buckets =>
      rowOf c tr (k / c.egress.buckets * c.egress.buckets + j))[j]! =
      rowOf c tr (k / c.egress.buckets * c.egress.buckets + j) :=
    getElem!_eq (by simp [hj])
  rw [hgrid]
  have htk : k / c.egress.buckets * c.egress.buckets + j ≤ k := by omega
  have hwin : (k / c.egress.buckets * c.egress.buckets + j) / c.egress.buckets = k / c.egress.buckets := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hT0, Nat.div_eq_of_lt hj, Nat.zero_add]
  have hwh := window_hands hE hc hst (n + 1) le_rfl
  have hWT : k / c.egress.buckets * c.egress.buckets + j <
      (pre c tr (n + 1)).shaper.syncs.size * c.egress.buckets := by
    have := Nat.mul_le_mul_right c.egress.buckets (show k / c.egress.buckets + 1 ≤ _ from hw)
    rw [Nat.succ_mul] at this
    omega
  generalize htd : k / c.egress.buckets * c.egress.buckets + j = t at htk hwin hWT ⊢
  have hwt : k / c.egress.buckets * c.egress.buckets ≤ t := by omega
  obtain ⟨m, b', hmn, hEm⟩ := egressAt_exists hE htk
  obtain ⟨now', he'⟩ := egressAt_get hEm
  have hm := egressAt_lt hEm
  have hgm := (prefix_good c tr m).1
  have htm := egressAt_t hEm
  have hth : (pre c tr m).shaper.t < horizon c := htm ▸ hEm.2.2
  have hcl := clean_of_complete hE hc hst hEm he' (htm ▸ htk) (htm ▸ hwin)
  have hr := egressMid_rel (pre c tr m) now' b'
  unfold Clean at hcl
  rw [hr.2.1] at hcl
  rw [rowOf_eq hEm, he']
  simp only [evRow]
  have hfi := fifo_inv c tr m hm.le
  have hrel := release hgm hfi hth hcl i hia
  rw [getElem!_eq hrel]
  have hf0 : (pre c tr m).shaper.certifier.fault = none := by
    rw [← hr.2.1] at hcl
    obtain ⟨-, -, f3, -⟩ := fifo_tick _ (hgm.shaper.rel hr)
    have h1 := (f3 hcl).1
    rw [← hr.1.of_none h1]
    exact h1
  have hi := hinv c tr (N := n + 1) (by omega) m (by omega)
  have hqq := hi.queue hf0
  have e : (tr.take (n + 1)).take (m + 1) = tr.take (m + 1) := by
    rw [List.take_take]; congr 1; omega
  have hfull : handed (init c) (tr.take (n + 1)) = handed (init c) (tr.take m) ++
      ((handedAt (pre c tr m) (.tick .egress now' b')).map (·, t) ++
        handed (pre c tr (m + 1)) ((tr.take (n + 1)).drop (m + 1))) := by
    conv_lhs => rw [← List.take_append_drop (m + 1) (tr.take (n + 1)), e]
    rw [handed_append, handed_take_succ c tr m hm, he', htm, List.append_assoc]
  have hrest : ∀ x ∈ handed (pre c tr (m + 1)) ((tr.take (n + 1)).drop (m + 1)), t + 1 ≤ x.2 := by
    have ht1 : (pre c tr (m + 1)).shaper.t = t + 1 := by
      rw [t_succ c tr m hm, he', htm]
      simp [run, tickView, htm ▸ hth]
    exact fun x hx => ht1 ▸ handed_ge (prefix_good c tr (m + 1)).1 _ x hx
  have hpast : ∀ y ∈ hlist c.egress.buckets (pre c tr (n + 1)).shaper.syncs (handed (init c) (tr.take m)), y < t :=
    forall₂_lt hi.le fun a ha => by
      obtain ⟨z, hz, rfl⟩ := List.mem_map.1 ha
      exact htm ▸ hfi.past z hz
  rw [hfull] at hwh
  rw [htm] at hqq ⊢
  have hle := hfi.le
  rw [hfull]
  set S := (pre c tr (n + 1)).shaper.syncs with hS
  change NetworkCertifier.expectedRow _ _ S.size (fun w => S[w]) _ _ _ = _
  generalize handed (init c) (tr.take m) = hp at hfull hpast hle hqq hwh ⊢
  generalize handedAt (pre c tr m) (.tick .egress now' b') = L at hfull hwh ⊢
  generalize handed (pre c tr (m + 1)) ((tr.take (n + 1)).drop (m + 1)) = R at hfull hrest hwh ⊢
  generalize (pre c tr m).shaper.queued = q at hle hqq ⊢
  generalize hT : c.egress.buckets = T at *
  have hmap : (hp ++ (L.map (·, t) ++ R)).map (fun x => logical x.1) =
      hp.map (fun x => logical x.1) ++ (L.map (·, t) ++ R).map (fun x => logical x.1) := List.map_append
  have hsplit : hlist T S (hp ++ (L.map (·, t) ++ R)) =
      hlist T S hp ++ hfrom T S (hlast T S 0 (hp.map fun x => logical x.1)) ((L.map (·, t) ++ R).map fun x => logical x.1) := by
    rw [hlist, hmap, hfrom_append]
  have hlen : (hlist T S hp).length = hp.length := by rw [hlist, hfrom_length, List.length_map]
  have hrest' := List.forall₂_drop hp.length hwh
  rw [hsplit, List.drop_left' hlen, show (hp ++ (L.map (·, t) ++ R)).map Prod.snd =
    hp.map Prod.snd ++ (L.map (·, t) ++ R).map Prod.snd from List.map_append, List.drop_left' (by simp)] at hrest'
  have hreq := forall₂_eq_of hrest' fun a b hab hb => hab (by
    rw [List.map_append] at hb
    rcases List.mem_append.1 hb with hb | hb
    · rw [List.map_map] at hb
      obtain ⟨x, -, rfl⟩ := List.mem_map.1 hb; exact hwt
    · obtain ⟨x, hx, rfl⟩ := List.mem_map.1 hb; have := hrest x hx; omega)
  have hcount : ∀ u ≤ t, NetworkCertifier.enqueued T S.size (fun w => S[w])
      ((hp ++ (L.map (·, t) ++ R)).map fun x => ⟨bitsOf x.1.bytes, x.1.anchor, x.1.index⟩) u =
      (hlist T S hp).count u + if t = u then L.length else 0 := by
    intro u hu
    unfold NetworkCertifier.enqueued
    simp only [show u < S.size * T by omega, ↓reduceIte]
    change (hlist T S (hp ++ (L.map (·, t) ++ R))).count u = _
    rw [hsplit, List.count_append, hreq]
    have hz : (R.map Prod.snd).count u = 0 :=
      List.count_eq_zero_of_not_mem fun hm => by
        obtain ⟨y, hy, e⟩ := List.mem_map.1 hm
        have := hrest y hy
        omega
    simp only [List.map_append, List.count_append, hz, List.map_map, Function.comp_def, List.map_const',
      List.count_replicate, beq_iff_eq, Nat.add_zero]
  unfold NetworkCertifier.expectedRow NetworkCertifier.fifoGrid NetworkCertifier.fill
  simp only [Fin.val_last]
  rw [offset_last]
  generalize NetworkCertifier.enqueued T S.size (fun w => S[w])
    ((hp ++ (L.map (·, t) ++ R)).map fun x => ⟨bitsOf x.1.bytes, x.1.anchor, x.1.index⟩) = E at hcount ⊢
  have hzero : (hlist T S hp).count t = 0 := List.count_eq_zero_of_not_mem fun h => by
    have := hpast t h
    omega
  have hlt : ∀ u < t, E u = (hlist T S hp).count u := fun u hu => by
    rw [hcount u hu.le]
    simp [Nat.ne_of_gt hu]
  have hEt : E t = L.length := by rw [hcount t le_rfl, hzero]; simp
  have hqE : NetworkCertifier.queue c.egress.allocation E t = q := by rw [queue_congr hlt, hqq]
  have hrelt : (NetworkCertifier.released c.egress.allocation E t : Nat) = min (q + L.length) c.egress.allocation := by
    simp only [NetworkCertifier.released]
    rw [hqE, hEt]
  have hoff : rsum (fun u => (NetworkCertifier.released c.egress.allocation E u : Nat)) t = hp.length - q := by
    have h1 := released_conserve c.egress.allocation E t
    rw [rsum_congr hlt, rsum_count _ t hpast, hqE, hlen] at h1
    omega
  rw [hrelt, hoff]
  split_ifs with h1
  · have hk : hp.length - q + i < (hp ++ L.map (·, t)).length := by simp; omega
    obtain ⟨v, hv⟩ : ∃ v, (hp ++ L.map (·, t))[hp.length - q + i]? = some v := ⟨_, List.getElem?_eq_getElem hk⟩
    rw [List.getElem?_map, hv]
    simp only [Option.map_some]
    unfold NetworkCertifier.payloads
    rw [List.getElem?_map, ← List.append_assoc, List.getElem?_append_left hk, hv]
    rfl
  · rfl

end Properties.CertifierDevice.Proofs
