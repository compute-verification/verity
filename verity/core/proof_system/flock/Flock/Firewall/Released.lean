/-
# What an accepted session lets out, under D8 (a)

With `holds_fixed`, the firewall's own values that leave an accepted session are the specified ones once
`FirewallComputes` holds: whatever the workers do, what leaves is honest output or nothing.
-/
import Flock.Firewall.Assumptions

namespace Flock.Firewall.Contract

/-- Under D8 (a), every value an accepted session lets out by the firewall's own computation is `spec`'s for its name. -/
theorem holds_computed {spec : String → Option ByteArray} {pub : PublicInput} {t : Transcript} (h : Holds pub t)
    (hc : FirewallComputes spec t) {what : String} {src : By} {want sent : ByteArray} {ps : Array Nat}
    (hm : Item.fixed what src want sent ps ∈ t.items.toList) (hs : src = .recheck ∨ src = .shadow ∨ src = .own) :
    spec what = some sent := by
  have e := ((holds_fixed h hm).2.2 (by rcases hs with rfl | rfl | rfl <;> decide)
    (by rcases hs with rfl | rfl | rfl <;> decide)).1
  rw [hc what src want sent ps hm hs]
  cases want
  cases sent
  simp_all

end Flock.Firewall.Contract
