import verity.circuits.Circuit

/-! The account, beside `account.py`. A placeholder that shows the protocol's Lean reading the circuit's. -/

namespace Verity.Protocol.Accounting

structure Execution where
  circuit : Verity.Circuits.Circuit
  device : String
  work : Nat

end Verity.Protocol.Accounting
