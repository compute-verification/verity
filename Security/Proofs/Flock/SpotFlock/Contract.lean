module

public import Proofs.Flock.SpotFlock.Numbers
public import Proofs.Flock.SamplingGeneric.TwoDeployed

/-!
# SPOT-Flock's two-stage contract at its deployed code

`SpotFlock.sessionsTwo_spot`: the two-stage compiler's contract (`CompiledTwo.Sessions`) at the deployed two-stage code
(`SpotFlock.code h`), Flock's masked record and SHA-512, the `sessions` of L3's two-stage refinement. It is
`SamplingDeploy.sessionsTwo_compiled` at the code's facts:

* the replay and second-stage samples, from the registration's match and A3 at the coins' bytes (`replayDraws_code`,
  `secondDraws_code`);
* the refusals and the interior roots (`refusesTwo_code`, `interiorFixed_code`);
* the sample's fixedness, from the round order (`drawFixed_codeTwo`);
* the sessions' runs and numbers (`sessionRuns`, `innerNumbers_code`).

It takes the sessions' openings at a replay block (`SessionOpensTwo`), the one premise left.
-/

@[expose] public section

namespace SpotFlock

variable (h : Held)

/-- **SPOT-Flock's two-stage contract at its deployed code**, from A3 at the coins' bytes and the sessions' openings. -/
theorem sessionsTwo_spot (hA : SamplingDeploy.CallsUniform (fun _ => L) fun _ => coinByte)
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (code h)) (two h) iroots) :
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
      (rs h)).Sessions FlockDeploy.sha512 :=
  SamplingDeploy.sessionsTwo_compiled (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h)
    (rs h) FlockDeploy.sha512
    (SamplingDeploy.replayDraws_code ((base h).atOrder (order h)) (reg h) (two h) (fun _ => L) (fun _ => L)
      (SamplingDeploy.bytesFirst (base h) (fun _ => L) (order h) fun _ => coinByte)
      (SamplingDeploy.bytesSecond (base h) (fun _ => L) (order h) fun _ => coinByte) (regMatchesTwo_code h)
      (SamplingDeploy.bytesUniform_atOrder (base h) (fun _ => L) (order h) (fun _ => coinByte) hA))
    (SamplingDeploy.secondDraws_code ((base h).atOrder (order h)) (reg h) (two h) (fun _ => L) (fun _ => L)
      (SamplingDeploy.bytesFirst (base h) (fun _ => L) (order h) fun _ => coinByte)
      (SamplingDeploy.bytesSecond (base h) (fun _ => L) (order h) fun _ => coinByte) (regMatchesTwo_code h)
      (SamplingDeploy.firstFixed_atOrder (base h) (fun _ => L) (order h) fun _ => coinByte)
      (SamplingDeploy.secondUniform_atOrder (base h) (fun _ => L) (order h) (fun _ => coinByte) hA))
    (refusesTwo_code h FlockDeploy.sha512) (interiorFixed_code h FlockDeploy.sha512)
    (SamplingDeploy.drawFixed_codeTwo ((base h).atOrder (order h)) (reg h) (fun _ => L) (fun _ => L)
      (SamplingDeploy.bytesFirst (base h) (fun _ => L) (order h) fun _ => coinByte)
      (SamplingDeploy.bytesSecond (base h) (fun _ => L) (order h) fun _ => coinByte)
      (SamplingDeploy.bytesFixedTwo_atOrder (base h) (fun _ => L) (fun _ => L) (order h) (fun _ => coinByte)
        fun _ => coinByte))
    (sessionRuns h _root_.Flock.Readable.relation FlockDeploy.sha512)
    (innerNumbers_code h _root_.Flock.Readable.relation) hopens

end SpotFlock
