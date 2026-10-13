import Models.Accounting.Guarantees.Memory
import Models.Accounting.Guarantees.Bandwidth
import Models.Accounting.Guarantees.Joint
import Models.Accounting.Guarantees.PerRun
import Models.Accounting.Guarantees.Pouw
import Models.Accounting.Guarantees.BandwidthAudit
import Models.Accounting.Guarantees.Capacity
import Models.Accounting.Guarantees.RateAudit
import Models.Accounting.Guarantees.RateAuditSketch
import Models.Accounting.Guarantees.Sketch
import Models.Accounting.Guarantees.Tag
import Models.Accounting.Guarantees.BoundReceipt
import Models.Accounting.Guarantees.BoundSketch

/-! Resource accounting's guarantee statements: the joint guarantee and its covert residual, PoUW's and the bandwidth
audit's guarantees as a joint run's compute and bandwidth guarantees, a state-bounded game's soundness as its capacity
guarantee, derived state's share of the capacity residual, the rate audit's residency and bandwidth with the rebuild
charged as compute and with a sketch-then-hash tag, the tag and sketch guarantees, co-location and the link relay
for receipts bound to the matmul, what a relay of the deployed bound receipt's sketch must exchange, the bandwidth
audit's guarantee, and their per-run forms. -/
