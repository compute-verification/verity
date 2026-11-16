module

public import Properties.Pouw.PearlPlus.Witnesses

/-! The key the test noise's F_A and F_B lines are sampled under (`Words.lineKey` of the test pass's seed_B). -/

@[expose] public section

namespace Properties.Pouw.PearlPlus

namespace Witness

/-- The key of B's E lines (`Words.lineKey` of seed_B). -/
def lineKeyB : ByteArray :=
  beBytes 32 [0xdec3da263f57eff0219219320eca345ad8292b1475b9b5c621d864115e2ece4e]

end Witness

end Properties.Pouw.PearlPlus
