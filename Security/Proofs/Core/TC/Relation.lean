import Definitions.Core.TC.Relation

/-! The proof-only part of `Definitions.Core.TC.Relation`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Verity.TC.Relation

/-- `relation.pack_state` on a canonical state (its fields are non-negative, except the exponent). -/
def packState (st : State) : Nat :=
  if st.nan ≠ 0 then 0x7FFFFFFF
  else if st.inf ≠ 0 then (st.neg.toNat <<< 31) ||| 0x7F800000
  else if st.M = 0 then 0
  else
    (st.neg.toNat <<< 31) ||| ((st.e + 127 - (if st.M.toNat >>> 23 ≠ 0 then 0 else 1)).toNat <<< 23)
      ||| (st.M.toNat &&& 0x7FFFFF)

end Verity.TC.Relation
