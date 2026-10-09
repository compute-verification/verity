//! The statement forms the soundness proof covers (verifier PROTOCOL.md §16.15): the Lean verifier's `Flock.ProvedScope.check`,
//! its rules in its order and its refusals word for word. The Lean verifier refuses every session of a statement outside
//! them, so a form no theorem covers is never accepted; the live server does the same ([`crate::StatementVerifier::refusal`]),
//! so its verdicts are the Lean verifier's.
//!
//! The live build writes only hm96-sha512 row statements with retained rounds and `hm96-sha512/v1` leaves, so the Lean
//! check's rules on those hold of every statement it sets up, as do its parse and its public file's load.

use crate::circuit::{port_cols, Composite, Instances, STATEMENT};
use crate::ir_block::IrUnitNet;

const WORD: usize = 128;

const UNCOVERED: &str = ": a statement form the soundness proof does not cover";

/// `ProvedScope.refusal`: the verdict's reason.
pub fn refusal(why: &str) -> String {
    format!("refused: {why}{UNCOVERED}")
}

/// Whether a verdict's reason is a [`refusal`].
pub fn is_refusal(why: &str) -> bool {
    why.starts_with("refused: ") && why.ends_with(UNCOVERED)
}

/// `ProvedScope.portReadsB`: every input column a computed row of `net` reads is a bit of one of its input ports.
pub fn port_reads(net: &IrUnitNet) -> bool {
    let n = (net.in_words * WORD).min(net.const_pos);
    let mut mask = vec![false; n];
    for (q, w) in port_cols(&net.in_groups) {
        for x in q..(q + w).min(n) {
            mask[x] = true;
        }
    }
    fn row(m: &[Vec<usize>], r: usize) -> &[usize] {
        m.get(r).map_or(&[], Vec::as_slice)
    }
    (n..net.const_pos).all(|r| row(&net.a, r).iter().chain(row(&net.b, r)).all(|&x| x >= n || mask[x]))
}

/// `ProvedScope.scopeWhy` of a circuit outside `scopeOk`, `None` inside: the unit reads only its leaf group, returns the
/// outputs, has at most `2^32` rows and reads only port bits among its inputs, and every row is a `sha512/row/v1` row.
pub fn circuit_why(c: &Composite) -> Option<String> {
    let unit = c.unit();
    if unit.in_groups.len() != 1 {
        Some(format!("a unit reading {} input groups", unit.in_groups.len()))
    } else if c.out_net != c.unit_net {
        Some("a circuit whose outputs a tail stage computes (META out_net)".into())
    } else if unit.unit_log > 32 {
        Some(format!("a unit of 2^{} rows", unit.unit_log))
    } else if !port_reads(unit) {
        Some("a unit row reading an input column that is not a port bit".into())
    } else if !c.ports.iter().all(|p| !p.v2 && p.segs.is_none() && p.bits == 16 * p.words && p.comps > 0) {
        Some("a row that is not a sha512/row/v1 row of 16 · words bits in at least one chunk".into())
    } else {
        None
    }
}

/// `ProvedScope.statementRules`' first failure among those a live statement can fail: `name` is the statement as the Lean
/// verifier names it, `typed` whether it is `verity/flock-circuit/types`, `tables` the session's tables and `public_inputs`
/// whether it has public input ports.
pub fn statement_why(name: &str, typed: bool, zk: bool, tables: usize, public_inputs: bool) -> Option<String> {
    if typed {
        Some(format!("{name}, a typed statement"))
    } else if !zk && tables > 1 {
        Some(format!("{tables} tables in a session without --zk"))
    } else if public_inputs {
        Some("public input ports (--public-inputs)".into())
    } else {
        None
    }
}

/// The statement's id as the Lean verifier names it: a selftest build's (`seed-injection`) is `+seed-injection`'s.
pub fn lean_name(c: &Composite) -> String {
    let base = if c.statement() == crate::typed::STATEMENT { crate::typed::STATEMENT } else { STATEMENT };
    format!("{base}{}", if cfg!(feature = "seed-injection") { "+seed-injection" } else { "" })
}

/// The shared-row rule of `ProvedScope.check` (`sharedRowsOk`): a shared-row public file (`shared`) is a covered form only in
/// a `--zk` session whose verifier holds its own registered reads (`registered`, `--registered F`). Every statement the
/// live build sets up has hidden outputs.
pub fn shared_rows_why(shared: bool, zk: bool, registered: bool) -> Option<&'static str> {
    (shared && !(zk && registered)).then_some("a shared-row public file (header shared_rows)")
}

/// **`ProvedScope.check`** on a statement the live build set up: its circuit, its public file (`inst`), `--zk` and its tables.
/// The Lean verifier holds its own `--registered F` exactly when the public header names registered reads: without it such
/// a header is refused at setup, and with it a header naming none is (`Registered.check`).
pub fn check(c: &Composite, inst: &Instances, zk: bool, tables: usize) -> Result<(), String> {
    let typed = c.statement() == crate::typed::STATEMENT;
    if let Some(why) = statement_why(&lean_name(c), typed, zk, tables, !c.public_inputs.is_empty()) {
        return Err(refusal(&why));
    }
    if let Some(why) = shared_rows_why(inst.tables.is_some(), zk, inst.header.get("registered").is_some()) {
        return Err(refusal(why));
    }
    circuit_why(c).map_or(Ok(()), |why| Err(refusal(&why)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_block::{FlatCache, Group};

    fn net(groups: Vec<Group>, in_words: usize, rows: Vec<(Vec<usize>, Vec<usize>)>) -> IrUnitNet {
        let const_pos = in_words * WORD + rows.len();
        let (mut a, mut b): (Vec<Vec<usize>>, Vec<Vec<usize>>) = (0..in_words * WORD).map(|i| (vec![i], vec![i])).unzip();
        for (ra, rb) in rows {
            a.push(ra);
            b.push(rb);
        }
        a.push(vec![const_pos]);
        b.push(vec![const_pos]);
        IrUnitNet { name: "unit".into(), useful: const_pos + 1, const_pos, in_groups: groups, out_groups: vec![], in_words, out_cols: vec![],
                    a, b, unit_log: 9, sha256: [0; 32], leaves: None, cut: None, flat: FlatCache::default() }
    }

    #[test]
    fn the_refusal_is_the_lean_verifiers() {
        assert_eq!(refusal("a unit reading 2 input groups"),
                   "refused: a unit reading 2 input groups: a statement form the soundness proof does not cover");
        assert!(is_refusal(&refusal("public input ports (--public-inputs)")));
        assert!(!is_refusal("R7: no proof for circuit/rep0 (lone rep)"));
        assert_eq!(statement_why("verity/flock-circuit/types+seed-injection", true, true, 2, true).unwrap(),
                   "verity/flock-circuit/types+seed-injection, a typed statement");
        assert_eq!(statement_why("verity/flock-circuit", false, false, 2, true).unwrap(), "2 tables in a session without --zk");
        assert_eq!(statement_why("verity/flock-circuit", false, true, 2, true).unwrap(), "public input ports (--public-inputs)");
        assert_eq!(statement_why("verity/flock-circuit", false, true, 2, false), None);
        assert_eq!(statement_why("verity/flock-circuit", false, false, 1, false), None);
    }

    #[test]
    fn a_shared_row_file_needs_zk_and_registered_reads() {
        let refused = Some("a shared-row public file (header shared_rows)");
        assert_eq!(shared_rows_why(true, true, true), None);
        assert_eq!(shared_rows_why(true, true, false), refused);
        assert_eq!(shared_rows_why(true, false, true), refused);
        assert_eq!(shared_rows_why(true, false, false), refused);
        assert_eq!(shared_rows_why(false, false, false), None);
        assert_eq!(refusal(refused.unwrap()),
                   "refused: a shared-row public file (header shared_rows): a statement form the soundness proof does not cover");
    }

    #[test]
    fn a_unit_row_reads_only_port_bits() {
        // one input word, its first 16 bits one port at column 0
        let port = || vec![Group { col: 0, words: 1, bits: vec![16] }];
        assert!(port_reads(&net(port(), 1, vec![(vec![0, 15], vec![3])])));
        // a computed row's own and later columns are not inputs
        assert!(port_reads(&net(port(), 1, vec![(vec![0], vec![128]), (vec![128], vec![129])])));
        assert!(!port_reads(&net(port(), 1, vec![(vec![16], vec![0])])));
        assert!(!port_reads(&net(port(), 1, vec![(vec![0], vec![0]), (vec![127], vec![128])])));
        // a port past the first word's bits
        let two = vec![Group { col: 0, words: 1, bits: vec![16, 16] }];
        assert!(port_reads(&net(two, 1, vec![(vec![31], vec![16])])));
    }
}
