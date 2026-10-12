/-! What the controls' pinned statement reads (`double`), and two listed escapes (`loop` and `loop'`, `partial def`s that
call each other: their compiled code mentions each other, which the replay must not take for a cycle). -/

namespace Control

def double (n : Nat) : Nat := 2 * n

mutual
partial def loop (n : Nat) : Nat := if n = 0 then 0 else loop' (n - 1)
partial def loop' (n : Nat) : Nat := if n = 0 then 0 else loop (n - 1)
end

end Control
