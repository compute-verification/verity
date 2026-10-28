import Definitions.Nci.Dark

/-!
# NCI's θ rule: the named premise

What the protocol lets a declaration's `θ` hold, a premise about the protocol rather than about any circuit. Every bound
through a containment, total or partial (`PartialContainment`), rests on it.
-/

namespace Nci.Assumptions

open Verity

/-- **θ is registered** ([THETA-REGISTERED]). The protocol's θ rule: every entry of a declaration's `θ` holds
registered information, the public data the registration log registers at the start of the deployment, and the
reference circuit `c` takes as its own `θ`, `c.θ`, exactly the inputs it reads of that kind. An output of an earlier
run, certified or not, is never in `θ`: registration happens at the start of a deployment and essentially never
changes. One registration log is one deployment, and `θ` is what the log's first entry registers: the first run's `θ`
cites its own entry, which opens the log, every later run's cites that same entry, and registering new public data
takes a new log, a new deployment the auditor sees as such. So no value computed off the books, before the run or
during it, is ever registered in `θ`, and neither is any novel input of `c`.

What the bounds use is the rule's consequence for a partial containment `M` of `c` in `d`: a gate that `M`'s computed
gates `S` read from outside `S` (`c.D.into M.S`: a dark gate of `c`, or an input gate of `c`) goes into `d`'s `θ` only
when it is in `c`'s `θ`. A dark value, or a novel input of `c`, that `d` listed in `θ` would be read for free, and an
IU could then thread `c`'s counted multiplications on it without paying for the read.

The NCI check (`NciCheck.Baseline`) enforces the provenance form: each `θ` entry cites the first entry of the
deployment's registration log (`theta-record-unknown`, `theta-not-at-deployment-start`), which is at or before every
run's own entry, and the root that entry registers the value under (`theta-root`), of the `registered` kind only
(`theta-provenance`), and the run's own entry holds the θ entry's parameter to that root (`theta-unbound`). What stays a
premise is what a check can't see, since commitments are hiding: that the data the log's first entry registers was
public, and not computed off the books before its registration (gradients, partial sums, updated weights on unmonitored
hardware), and that the auditor knows which log is the deployment's. So this is a premise of every guarantee through a
containment: a partial one, or a total one `M : Containment c.D d.D` as the partial containment with no dark gates,
`PartialContainment.ofContainment M`, for which it says that no novel input of `c` that `c` reads is in `d`'s `θ`. -/
def ThetaRegistered (c d : Registered) (M : PartialContainment c d) : Prop :=
  ∀ a ∈ c.D.into M.S, M.emb a ∈ d.θ → a ∈ c.θ

end Nci.Assumptions
