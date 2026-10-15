//! The statement forms the soundness proof covers (verifier PROTOCOL.md §16.15): the Lean verifier's `Flock.ProvedScope.check`,
//! its rules in its order and its refusals word for word. The Lean verifier refuses every session of a statement outside
//! them, so a form no theorem covers is never accepted; the live server does the same ([`crate::StatementVerifier::refusal`]),
//! so its verdicts are the Lean verifier's.
//!
//! The live build writes only hm96-sha512 row statements with retained rounds and `hm96-sha512/v1` leaves, so the Lean
//! check's rules on those hold of every statement it sets up, as do its parse and its public file's load. Its statements
//! all have hidden outputs, so the circuit's check is `scopeOkH`.

use crate::circuit::{port_cols, seg_shape, Composite, Instances, Port, STATEMENT};
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

/// `ProvedScope.v1Ports`' row: a `sha512/row/v1` row of `16 · words` bits in at least one chunk.
fn v1_row(p: &Port) -> bool {
    !p.v2 && p.segs.is_none() && p.bits == 16 * p.words && p.comps > 0
}

/// `ProvedScope.v2Row`: an unsegmented `sha512/row/v2` row of `16 · words` bits, its bytes filling its compressions' blocks
/// and ending, with SHA-512's 17 bytes of padding, in its hm96 slot's block.
fn v2_row(p: &Port) -> bool {
    p.v2 && p.segs.is_none() && p.bits == 16 * p.words && WORD * p.comps <= 2 * p.words && 2 * p.words + 17 <= WORD * (p.comps + 1)
}

/// `ProvedScope.segRow`: a `sha512/row-seg/v1` row segmented as [`seg_shape`] cuts its bits, of `16 · words` bits, its
/// compressions its segments' and then its outer message's, and each segment's bytes ending, with SHA-512's 17 bytes of
/// padding, in its last compression's block.
fn seg_row(p: &Port) -> bool {
    let (segs, outer) = seg_shape(p.bits);
    p.segs.as_ref() == Some(&segs) && p.bits == 16 * p.words && p.comps == segs.iter().map(|s| s.1).sum::<usize>() + outer - 1 &&
        segs.iter().all(|&(m, c)| WORD * (c - 1) <= m.div_ceil(8))
}

/// `ProvedScope.bitRows`: every row a [`v2_row`] or a [`seg_row`], in any mix, one output port each, the output rows (the
/// last `out_row_ports.len()`) `v2_row`s.
fn bit_rows(ports: &[Port], out_row_ports: &[usize]) -> bool {
    let n_in = ports.len().saturating_sub(out_row_ports.len());
    ports.iter().all(|p| v2_row(p) || seg_row(p)) && out_row_ports.iter().all(|&k| k == 1) &&
        (0..out_row_ports.len()).all(|j| ports.get(n_in + j).is_some_and(v2_row))
}

/// `ProvedScope.scopeWhy hidden` of a circuit outside `scopeOkH` (`scopeOk` when not `hidden`), `None` inside: the unit
/// reads only its leaf group, returns the outputs, has at most `2^32` rows and reads only port bits among its inputs, and
/// every row is a `sha512/row/v1` row, or, with hidden outputs, its rows are [`bit_rows`].
pub fn circuit_why(c: &Composite, hidden: bool) -> Option<String> {
    let unit = c.unit();
    if unit.in_groups.len() != 1 {
        Some(format!("a unit reading {} input groups", unit.in_groups.len()))
    } else if c.out_net != c.unit_net {
        Some("a circuit whose outputs a tail stage computes (META out_net)".into())
    } else if unit.unit_log > 32 {
        Some(format!("a unit of 2^{} rows", unit.unit_log))
    } else if !port_reads(unit) {
        Some("a unit row reading an input column that is not a port bit".into())
    } else {
        rows_why(&c.ports, &c.out_row_ports, hidden).map(String::from)
    }
}

/// `ProvedScope.scopeWhy hidden`'s last rule: the rows `v1Ports` (or, with hidden outputs, `bitRows`) admits.
fn rows_why(ports: &[Port], out_row_ports: &[usize], hidden: bool) -> Option<&'static str> {
    let v1 = ports.iter().all(v1_row);
    if !hidden && !v1 {
        Some("a row that is not a sha512/row/v1 row of 16 · words bits in at least one chunk")
    } else if hidden && !v1 && !bit_rows(ports, out_row_ports) {
        Some("rows that are neither all sha512/row/v1 rows of 16 · words bits in at least one chunk nor all unsegmented \
              sha512/row/v2 rows of 16 · words bits ending in their hm96 block or sha512/row-seg/v1 rows of 16 · words \
              bits whose segments end in their last block, the output rows unsegmented, one output port each")
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
    circuit_why(c, true).map_or(Ok(()), |why| Err(refusal(&why)))
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

    fn port(words: usize, comps: usize, bits: usize, v2: bool, segs: Option<Vec<(usize, usize)>>) -> Port {
        Port { name: "x".into(), words, comps, bits, v2, segs }
    }

    #[test]
    fn rows_in_scope_are_lean_v1_or_v2_rows() {
        assert!(v1_row(&port(64, 1, 1024, false, None)));
        assert!(!v1_row(&port(64, 0, 1024, false, None)));
        assert!(!v1_row(&port(64, 1, 1024, true, None)));
        // a v2 row of 64 or 100 words ends, padded, in its slot's block; at 120 its last 112 bytes leave no room for the padding
        assert!(v2_row(&port(64, 1, 1024, true, None)));
        assert!(v2_row(&port(100, 1, 1600, true, None)));
        // 32 words' 64 bytes and the padding fit the slot's block alone
        assert!(v2_row(&port(32, 0, 512, true, None)));
        assert!(!v2_row(&port(120, 1, 1920, true, None)));
        assert!(!v2_row(&port(32, 1, 512, true, None)));
        assert!(!v2_row(&port(64, 1, 1000, true, None)));
        assert!(!v2_row(&port(64, 1, 1024, true, Some(vec![(1024, 1)]))));
        assert!(!v2_row(&port(64, 1, 1024, false, None)));
    }

    #[test]
    fn only_hidden_outputs_admit_v2_rows() {
        let v1 = [port(64, 1, 1024, false, None)];
        let v2 = [port(64, 1, 1024, true, None), port(32, 0, 512, true, None)];
        assert_eq!(rows_why(&v1, &[1], false), None);
        assert_eq!(rows_why(&v1, &[2], true), None);
        assert_eq!(rows_why(&v2, &[1], true), None);
        assert_eq!(rows_why(&v2, &[1], false), Some("a row that is not a sha512/row/v1 row of 16 · words bits in at least one chunk"));
        assert!(rows_why(&v2, &[2], true).unwrap().starts_with("rows that are neither"));
    }

    fn seg(bits: usize, comps: usize) -> Port {
        port(bits / 16, comps, bits, true, Some(seg_shape(bits).0))
    }

    #[test]
    fn rows_in_scope_are_lean_seg_rows() {
        // one segment of two compressions, the outer message's digest and padding in the last; three segments (8192, 8192,
        // 1024 bits: 9 + 9 + 2 compressions) and the outer message's 192 bytes in one more
        assert!(seg_row(&seg(1024, 2)));
        assert!(seg_row(&seg(17408, 21)));
        assert_eq!(seg_shape(17408).0, vec![(8192, 9), (8192, 9), (1024, 2)]);
        assert!(!seg_row(&seg(17408, 20)));
        // 896 bits: the segment's 112 bytes and the padding spill past its last block
        assert!(!seg_row(&seg(896, 2)));
        assert!(!seg_row(&port(64, 2, 1024, true, Some(vec![(1024, 1)]))));
        assert!(!seg_row(&port(64, 2, 1024, true, None)));
        assert!(!seg_row(&port(64, 2, 1000, true, Some(seg_shape(1000).0))));
        assert!(!v2_row(&seg(1024, 2)));
    }

    #[test]
    fn hidden_outputs_admit_seg_rows_alone_or_mixed_the_outputs_unsegmented() {
        let out = port(64, 1, 1024, true, None);
        // Lean's witnessSeg and witnessMix (ZkHidden/NonVacuousSeg.lean)
        assert_eq!(rows_why(&[seg(1024, 2), out.clone()], &[1], true), None);
        assert_eq!(rows_why(&[seg(17408, 21), port(64, 1, 1024, true, None), out.clone()], &[1], true), None);
        assert!(rows_why(&[seg(1024, 2), out.clone()], &[1], false).is_some());
        // witnessSegOut, witnessSegLate, two output ports, a v1 row among them
        let why = |ports: &[Port], outs: &[usize]| rows_why(ports, outs, true).unwrap();
        assert_eq!(why(&[seg(1024, 2), seg(1024, 2)], &[1]),
                   "rows that are neither all sha512/row/v1 rows of 16 · words bits in at least one chunk nor all unsegmented \
                    sha512/row/v2 rows of 16 · words bits ending in their hm96 block or sha512/row-seg/v1 rows of 16 · words \
                    bits whose segments end in their last block, the output rows unsegmented, one output port each");
        assert!(rows_why(&[seg(896, 2), out.clone()], &[1], true).is_some());
        assert!(rows_why(&[seg(1024, 2), out.clone()], &[2], true).is_some());
        assert!(rows_why(&[seg(1024, 2), port(64, 1, 1024, false, None), out.clone()], &[1], true).is_some());
        assert!(rows_why(&[seg(1024, 2)], &[1], true).is_some());
        assert_eq!(rows_why(&[seg(1024, 2), out.clone(), out], &[1, 1], true), None);
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
