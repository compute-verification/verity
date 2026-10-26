import Definitions.Accounting.RateAuditInteractive
import Definitions.Accounting.Bandwidth.Interactive

/-!
# The audits' devices as core's adversaries

Each device's tags and ingress are its adversary's output and ingress (`Verity.Adversary.output`, `.ingress`), at the
window's coins.
-/

namespace Accounting.Rate

open Pous (Bits)

variable {N b h ρ S L q : ℕ}

theorem Device.tags_eq_output (A : Device N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.tags E R O = Verity.Adversary.output A E (R, O) := rfl

theorem Device.ingress_eq (A : Device N b h ρ S L q) (E : Store N b) (R : Bits ρ) (O : Oracle N b h ρ) :
    A.ingress E R O = Verity.Adversary.ingress A E (R, O) := rfl

theorem InteractiveDevice.tags_eq_output (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : A.tags E R O = Verity.Adversary.output A E (R, O) := rfl

theorem InteractiveDevice.ingress_eq (A : InteractiveDevice N b h ρ S L q) (E : Store N b) (R : Bits ρ)
    (O : Oracle N b h ρ) : A.ingress E R O = Verity.Adversary.ingress A E (R, O) := rfl

end Accounting.Rate

namespace Bandwidth

open Pous (Bits)

variable {N b h S c G q : ℕ}

theorem Device.tags_eq_output (A : Device N b h S c G q) (E : Store N b) (O : Oracle N b h) :
    A.tags E O = Verity.Adversary.output A E O := rfl

theorem Device.ingress_eq (A : Device N b h S c G q) (E : Store N b) (O : Oracle N b h) :
    A.ingress E O = Verity.Adversary.ingress A E O := rfl

theorem InteractiveDevice.tags_eq_output (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) :
    A.tags E O = Verity.Adversary.output A E O := rfl

theorem InteractiveDevice.ingress_eq (A : InteractiveDevice N b h S c G q) (E : Store N b) (O : Oracle N b h) :
    A.ingress E O = Verity.Adversary.ingress A E O := rfl

end Bandwidth
