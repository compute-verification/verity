module

public import Proofs.Properties.Flock.Nets.Twin
public import Proofs.Flock.Soundness.Discharge.Hm.Top

/-!
# A compression's run, chunk by chunk

A `Cert` says where a compression's run puts each committed word: every schedule word from 16 on, and each round's new
`a` and `e`, is 64 fresh bits in a row, so the run's state after every 16 schedule words and every 8 rounds is a closed
term of the certificate's numbers. `schedOk`, `roundOk` and `ffOk` check one chunk each by running the twin from that
state: its result, its next bit, and that every row it adds reads at most 128 bits.
-/

@[expose] public section

open Flock Flock.HmNets FlockSoundness.Discharge.Hm

namespace FlockDeploy.Concrete.Nets

/-- A compression's run: it starts at bit `n0`; schedule word `16 + i` is bits `S[i] …`, round `t`'s new `a` and `e` bits
`A[t] …` and `E[t] …`, and the feed-forward ends at `fin`. -/
structure Cert where
  n0 : Nat
  S : List Nat
  A : List Nat
  E : List Nat
  fin : Nat

/-- The schedule after `16 j` more words. -/
def schedAt (msg : List Word) (c : Cert) (j : Nat) : List Word := msg ++ (c.S.take (16 * j)).map (vars · 64)

/-- The next bit after `16 j` schedule words. -/
def schedNext (c : Cert) (j : Nat) : Nat := if j = 0 then c.n0 else c.S.getD (16 * j - 1) 0 + 64

/-- The registers after `t` rounds. -/
def regsAt (cv : List Word) (c : Cert) (t : Nat) : Regs Word :=
  if t = 0 then Regs.ofList cv
  else ⟨vars (c.A.getD (t - 1) 0) 64, vars (c.A.getD (t - 2) 0) 64, vars (c.A.getD (t - 3) 0) 64,
    vars (c.A.getD (t - 4) 0) 64, vars (c.E.getD (t - 1) 0) 64, vars (c.E.getD (t - 2) 0) 64,
    vars (c.E.getD (t - 3) 0) 64, vars (c.E.getD (t - 4) 0) 64⟩

/-- The next bit after `t` rounds. -/
def roundNext (c : Cert) (t : Nat) : Nat := if t = 0 then schedNext c 4 else c.A.getD (t - 1) 0 + 64

/-- Every row reads at most `W` bits. -/
def widthOk (W : Nat) (rows : List Row) : Bool := rows.all fun r => Nat.ble (r.a.s.length + r.b.s.length) W

/-- Schedule words `16 j + 16 … 16 j + 31`. -/
noncomputable def schedOk (msg : List Word) (c : Cert) (j : Nat) : Bool :=
  let p := (scheduleK 16 (schedAt msg c j)).run ⟨schedNext c j, []⟩
  decide (p.1 = schedAt msg c (j + 1)) && p.2.next == schedNext c (j + 1) && widthOk 128 p.2.rows

/-- Rounds `8 j … 8 j + 7`. -/
noncomputable def roundOk (cv msg : List Word) (c : Cert) (j : Nat) : Bool :=
  let p := (roundsK (schedAt msg c 4) 8 (8 * j) (regsAt cv c (8 * j))).run ⟨roundNext c (8 * j), []⟩
  decide (p.1.toList = (regsAt cv c (8 * j + 8)).toList) && p.2.next == roundNext c (8 * j + 8) &&
    widthOk 128 p.2.rows

/-- The feed-forward: it ends at `fin`, and each output bit reads at most 16 bits. -/
noncomputable def ffOk (cv : List Word) (c : Cert) : Bool :=
  let p := (ffK cv (regsAt cv c 80)).run ⟨roundNext c 80, []⟩
  p.2.next == c.fin && widthOk 128 p.2.rows && p.1.all fun w => w.all fun f => Nat.ble f.s.length 16

end FlockDeploy.Concrete.Nets
