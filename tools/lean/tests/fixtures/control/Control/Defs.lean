/-! What the controls' pinned statement reads (`double`), and one listed escape (`loop`, a `partial def`). -/

namespace Control

def double (n : Nat) : Nat := 2 * n

partial def loop (n : Nat) : Nat := if n = 0 then 0 else loop (n - 1)

end Control
