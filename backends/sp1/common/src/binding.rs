//! Binding the typed object to the statement a proof commits (lane IB),
//! mirroring `veritor/interface/binding.py`.
//!
//! The guest proves a lowered statement `L` and commits `h_L = sha256(statement
//! bytes)`; when handed a typed object digest it also carries `h_O` -- the
//! canonical digest of the `typed-obligation/v0` object `L` was lowered from
//! -- through its public values *unchecked* (`public_values_with_object`).
//! The lowering erases fields of the object (reference naming, targets, the
//! declared interface and its projection; `lean/VeritorSpec/IRErasure.lean`,
//! `lower_eq_iff`), so two objects can share one `L`; and a prover could pair
//! `H(O_1)` with an unrelated `L_2`.  The fence (IR §I.3, GUIDANCE §17.3):
//!
//! ```text
//! typed object O  --validate-->  h_O  --deterministic lower-->  L, h_L
//! public values = h_O || h_L || verdict                    (65 bytes)
//! verifier: h_L == digest(lower(O))  for the O with digest h_O
//! ```
//!
//! [`check_binding`] is the verifier-side comparison.  The caller supplies
//! `h_O` and `h_L` *as it computed them* from the object it holds -- by
//! validating it and lowering it (the conformance test's `validate` + `lower`,
//! or the Python reference) -- and this function compares them with the
//! committed bytes.  Validation of the object is therefore the caller's, and
//! is not optional: an object that does not validate has no lowering and
//! nothing to bind (the Python reference reports that as `E_OBJECT_INVALID`
//! with the validator's code).  The check establishes identity -- the
//! statement the guest verified is the lowering of this object and of no
//! object with another digest -- and nothing else: not plan validity (RA's
//! validator), not that the object is the canonical derivation for a
//! challenge (GUIDANCE §17.4).
//!
//! No allocation, no hashing: byte comparison on fixed layouts.

use crate::{public_values_with_object, PUBLIC_VALUES_LEN, PUBLIC_VALUES_WITH_OBJECT_LEN};

/// Why the binding check fails, in the order the check tries them.  The names
/// are the Python reference's codes (`BindingError::name`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BindingError {
    /// 33 public bytes: the proof commits `h_L || verdict` only and binds no
    /// typed object.  Refused even when `h_L` matches: nothing says *which*
    /// object lowered to the statement.
    UnboundObject,
    /// Neither 33 nor 65 public bytes.
    PublicValuesLength(usize),
    /// The first 32 bytes are not the caller's `h_O`.
    ObjectDigestMismatch,
    /// Bytes `[32, 64)` are not the caller's `h_L`: the statement the guest
    /// verified is not the lowering of this object.
    LoweringMismatch,
}

impl BindingError {
    /// The Python reference's code for this error.
    pub fn name(self) -> &'static str {
        match self {
            BindingError::UnboundObject => "E_UNBOUND_OBJECT",
            BindingError::PublicValuesLength(_) => "E_PUBLIC_VALUES_LENGTH",
            BindingError::ObjectDigestMismatch => "E_OBJECT_DIGEST_MISMATCH",
            BindingError::LoweringMismatch => "E_LOWERING_MISMATCH",
        }
    }
}

/// The 65 public bytes a correct prover produces for an object with digest
/// `h_o` whose lowering has digest `h_l`: `h_o || h_l || 1` -- what
/// [`check_binding`] accepts for that object with the accept verdict.
pub fn bound_public_values(h_o: &[u8; 32], h_l: &[u8; 32]) -> [u8; PUBLIC_VALUES_WITH_OBJECT_LEN] {
    public_values_with_object(h_o, h_l, true)
}

/// Check that `public_values` bind the object with digest `h_o` to its own
/// lowering, whose digest is `h_l`; fail-closed.
///
/// `public_values` must be exactly 65 bytes `h_O' || h_L' || verdict`
/// ([`BindingError::UnboundObject`] for the 33-byte layout,
/// [`BindingError::PublicValuesLength`] otherwise); then `h_O' == h_o`
/// ([`BindingError::ObjectDigestMismatch`]) and `h_L' == h_l`
/// ([`BindingError::LoweringMismatch`] -- the checked lowering relationship).
/// Only when both hold is the verdict byte returned, as committed (`1` / `0`,
/// or a diagnostic build's `0xEE` / `0xEF`: decoding it is
/// [`crate::decode_verdict`]'s).
pub fn check_binding(h_o: &[u8; 32], h_l: &[u8; 32], public_values: &[u8]) -> Result<u8, BindingError> {
    match public_values.len() {
        PUBLIC_VALUES_WITH_OBJECT_LEN => {}
        PUBLIC_VALUES_LEN => return Err(BindingError::UnboundObject),
        other => return Err(BindingError::PublicValuesLength(other)),
    }
    if public_values[..32] != h_o[..] {
        return Err(BindingError::ObjectDigestMismatch);
    }
    if public_values[32..64] != h_l[..] {
        return Err(BindingError::LoweringMismatch);
    }
    Ok(public_values[64])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_and_codes() {
        let h_o = [0x11u8; 32];
        let h_l = [0x22u8; 32];
        let other = [0x33u8; 32];
        let correct = bound_public_values(&h_o, &h_l);
        assert_eq!(correct.len(), 65);
        assert_eq!(&correct[..32], &h_o);
        assert_eq!(&correct[32..64], &h_l);
        assert_eq!(check_binding(&h_o, &h_l, &correct), Ok(1));
        assert_eq!(check_binding(&h_o, &h_l, &public_values_with_object(&h_o, &h_l, false)), Ok(0));
        assert_eq!(check_binding(&h_o, &h_l, &correct[32..]), Err(BindingError::UnboundObject));
        assert_eq!(check_binding(&h_o, &h_l, &correct[..64]), Err(BindingError::PublicValuesLength(64)));
        assert_eq!(check_binding(&h_o, &h_l, &[]), Err(BindingError::PublicValuesLength(0)));
        assert_eq!(
            check_binding(&h_o, &h_l, &bound_public_values(&other, &h_l)),
            Err(BindingError::ObjectDigestMismatch)
        );
        assert_eq!(
            check_binding(&h_o, &h_l, &bound_public_values(&h_o, &other)),
            Err(BindingError::LoweringMismatch)
        );
        // both wrong: the object digest is compared first, as in the Python reference
        assert_eq!(
            check_binding(&h_o, &h_l, &bound_public_values(&other, &other)),
            Err(BindingError::ObjectDigestMismatch)
        );
        assert_eq!(BindingError::LoweringMismatch.name(), "E_LOWERING_MISMATCH");
    }
}
