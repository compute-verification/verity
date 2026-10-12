import Models.Accounting.Guarantees.Joint
import Models.Accounting.Guarantees.Pouw
import Models.Accounting.Guarantees.BandwidthAudit
import Models.Accounting.Guarantees.Capacity
import Models.Accounting.Guarantees.RateAudit
import Models.Accounting.Guarantees.RateAuditRemote
import Models.Accounting.Guarantees.RateAuditHorizon
import Models.Accounting.Guarantees.RateAuditNamed
import Models.Accounting.Guarantees.RateAuditSketch
import Models.Accounting.Guarantees.Sketch
import Models.Accounting.Guarantees.Tag

/-! Resource accounting's guarantee statements: the joint guarantee and its covert residual, PoUW's and the bandwidth
audit's guarantees as a joint run's compute and bandwidth guarantees, a state-bounded game's soundness as its capacity
guarantee, derived state's share of the capacity residual, the rate audit's residency and bandwidth with the rebuild
charged as compute and with a sketch-then-hash tag, the remote rate audit's conservation over tiers and chained
passes, the causal horizon bound on its samples and its conservation with named words, and the tag and sketch guarantees. -/
