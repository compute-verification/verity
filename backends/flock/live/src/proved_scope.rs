//! The statement forms the soundness proof covers (verifier PROTOCOL.md §16.15, §16.16): the Lean verifier's `Flock.ProvedScope.check`,
//! its rules in its order and its refusals word for word. The Lean verifier refuses every session of a statement outside
//! them, so a form no theorem covers is never accepted; the live server does the same ([`crate::StatementVerifier::refusal`]),
//! so its verdicts are the Lean verifier's.
//!
//! The live build writes only hm96-sha512 row statements with retained rounds and `hm96-sha512/v1` leaves, so the Lean
//! check's rules on those hold of every statement it sets up, as do its parse and its public file's load.

use crate::circuit::{port_cols, Composite, Instances, Port, STATEMENT};
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

/// `ProvedScope.rowOk`: a canonical row (PROTOCOL.md §16.16), a `sha512/row/v2` bit row, unsegmented.
pub fn row_ok(p: &Port) -> bool {
    p.v2 && p.segs.is_none()
}

/// Whether a row is a bit row, `sha512/row/v2` or segmented: the Lean verifier's `Port.v2`.
fn bit_row(p: &Port) -> bool {
    p.v2 || p.segs.is_some()
}

/// `ProvedScope.outRowsExact`: every output row holds exactly the bits of its run of output ports (no zero-padded output row).
pub fn out_rows_exact(c: &Composite) -> bool {
    c.out_rows().iter().enumerate().all(|(j, p)| p.bits == 16 * c.out_group(j).3)
}

/// `ProvedScope.cutsCovered`: every unit's every cut input (a port of `unit` past its leaf group's) is a leaf cut, a row
/// word, among `leaf_cuts` (`(unit, input port, leaf)`), for the `upv` units of a VU: none is wired from another net.
pub fn cuts_covered(unit: &IrUnitNet, upv: usize, leaf_cuts: &[(usize, usize, usize)]) -> bool {
    let n0 = unit.in_groups.first().map_or(0, |g| g.bits.len());
    let np = port_cols(&unit.in_groups).len();
    let mut cut = vec![false; upv * np];
    for &(u, j, _) in leaf_cuts {
        if u < upv && j < np {
            cut[u * np + j] = true;
        }
    }
    (0..upv).all(|k| (n0..np).all(|p| cut[k * np + p]))
}

/// `ProvedScope.canonicalWhy` of a circuit outside the canonical format the canonical headline covers, `None` inside
/// (`ProvedScope.canonicalOk` under `--zk`, every live statement having hidden outputs): the unit returns the outputs, every
/// row is canonical, output rows hold exactly their ports' bits, every unit input is a row bit or the zero, computed unit
/// rows read only port bits, the unit has at most `2^32` rows, and the VUs per block are a power of two.
pub fn canonical_why(c: &Composite) -> Option<String> {
    let unit = c.unit();
    if c.out_net != c.unit_net {
        Some("a circuit whose outputs a tail stage computes (META out_net)".into())
    } else if !c.ports.iter().all(row_ok) {
        Some("a row that is not an unsegmented sha512/row/v2 row, in a circuit with sha512/row/v2 rows".into())
    } else if !out_rows_exact(c) {
        Some("an output row that is not exactly its output ports' bits".into())
    } else if !cuts_covered(unit, c.upv, &c.leaf_cuts) {
        Some("a unit input wired from another net's output, not a row bit".into())
    } else if !port_reads(unit) {
        Some("a unit row reading an input column that is not a port bit".into())
    } else if unit.unit_log > 32 {
        Some(format!("a unit of 2^{} rows", unit.unit_log))
    } else if !c.g.is_power_of_two() {
        Some("a vus_per_block that is not a power of two".into())
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

/// **`ProvedScope.check`** on a statement the live build set up: its circuit, its public file (`inst`), `--zk` and its tables.
/// Its circuit is in `scopeOk` or, under `--zk`, in the canonical format (`canonicalOk`: every live statement has hidden
/// outputs); a circuit in neither is refused with `canonical_why` when it has a bit row under `--zk`, else `circuit_why`.
pub fn check(c: &Composite, inst: &Instances, zk: bool, tables: usize) -> Result<(), String> {
    let typed = c.statement() == crate::typed::STATEMENT;
    if let Some(why) = statement_why(&lean_name(c), typed, zk, tables, !c.public_inputs.is_empty()) {
        return Err(refusal(&why));
    }
    if inst.tables.is_some() {
        return Err(refusal("a shared-row public file (header shared_rows)"));
    }
    let Some(why) = circuit_why(c) else { return Ok(()) };
    if !zk {
        return Err(refusal(&why));
    }
    match canonical_why(c) {
        None => Ok(()),
        Some(cw) => Err(refusal(if c.ports.iter().any(bit_row) { &cw } else { &why })),
    }
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

    #[test]
    fn every_cut_input_is_a_leaf_cut() {
        // a leaf group of one port (port 0) and a cut group of two (ports 1 and 2)
        let groups = || vec![Group { col: 0, words: 1, bits: vec![16] }, Group { col: 1, words: 1, bits: vec![16, 16] }];
        let unit = net(groups(), 2, vec![]);
        assert!(cuts_covered(&unit, 2, &[(0, 1, 0), (0, 2, 0), (1, 1, 0), (1, 2, 3)]));
        // unit 1's port 2 has no leaf cut: a wire feeds it
        assert!(!cuts_covered(&unit, 2, &[(0, 1, 0), (0, 2, 0), (1, 1, 0)]));
        // a leaf cut at a leaf port, past the ports or past the units covers nothing
        assert!(!cuts_covered(&unit, 1, &[(0, 0, 0), (0, 3, 0), (0, 1, 0), (1, 2, 0)]));
        // a unit reading its leaf group alone has no cut input
        assert!(cuts_covered(&net(vec![Group { col: 0, words: 1, bits: vec![16] }], 1, vec![]), 4, &[]));
    }

    /// `check` accepts a circuit in the canonical format under `--zk` alone, and refuses each other form with the Lean
    /// verifier's reason: without `--zk` the v1-row rule's, under it the first canonical conjunct the circuit fails (an
    /// output row padded past its port's word, a segmented row, VUs per block not a power of two, a unit past `2^32` rows),
    /// and before either a tail, public input ports (under `--zk` too, in the canonical format) and a shared-row public file.
    #[test]
    fn the_canonical_format_is_accepted_under_zk_alone() {
        use crate::circuit::tests::{flat_tail_statement, flat_v2_statement};
        use crate::circuit::RowTables;

        let mut inst = Instances { header: serde_json::Value::Null, n: 0, rows: None, salts: None, bcs: vec![], public_sha512: [0; 64],
                                   population: None, tables: None, public: vec![] };
        let parse = |text: String| Composite::parse(&text, None).unwrap_or_else(|e| panic!("{e}"));
        let c = parse(flat_v2_statement(16, false));
        assert_eq!(canonical_why(&c), None);
        assert_eq!(check(&c, &inst, true, 1), Ok(()));
        assert_eq!(check(&c, &inst, true, 2), Ok(()));
        let v1 = "a row that is not a sha512/row/v1 row of 16 · words bits in at least one chunk";
        assert_eq!(check(&c, &inst, false, 1), Err(refusal(v1)));
        let refused = |c: &Composite, inst: &Instances| check(c, inst, true, 1).unwrap_err();
        assert_eq!(refused(&parse(flat_v2_statement(1024, false)), &inst),
                   refusal("an output row that is not exactly its output ports' bits"));
        assert_eq!(refused(&parse(flat_v2_statement(16, true)), &inst),
                   refusal("a row that is not an unsegmented sha512/row/v2 row, in a circuit with sha512/row/v2 rows"));
        let mut g3 = c.clone();
        g3.g = 3;
        assert_eq!(refused(&g3, &inst), refusal("a vus_per_block that is not a power of two"));
        let mut big = c.clone();
        big.nets[big.unit_net].1.unit_log = 33;
        assert_eq!(refused(&big, &inst), refusal("a unit of 2^33 rows"));
        // a tail of v1 rows: no bit row, so the scope rule's reason, which is also the canonical format's first
        let tail = parse(flat_tail_statement());
        assert_eq!(canonical_why(&tail).as_deref(), Some("a circuit whose outputs a tail stage computes (META out_net)"));
        assert_eq!(refused(&tail, &inst), refusal("a circuit whose outputs a tail stage computes (META out_net)"));
        let mut public = c.clone();
        public.public_inputs = vec![("x".into(), 64)];
        assert_eq!(refused(&public, &inst), refusal("public input ports (--public-inputs)"));
        inst.tables = Some(RowTables { counts: vec![1, 1], rows: None, salts: None, refs: vec![] });
        assert_eq!(refused(&c, &inst), refusal("a shared-row public file (header shared_rows)"));
        // public input ports are a statement rule's, before the public file's (test_lean_canonical_format's
        // `test_a_canonical_statement_with_a_public_input_port_is_refused_for_its_ports`)
        assert_eq!(refused(&public, &inst), refusal("public input ports (--public-inputs)"));
    }

    #[test]
    fn a_canonical_row_is_an_unsegmented_bit_row() {
        let p = |v2, segs: Option<Vec<(usize, usize)>>| Port { name: "p".into(), words: 2, comps: 1, bits: 20, v2, segs };
        assert!(row_ok(&p(true, None)));
        assert!(!row_ok(&p(false, None)));
        // the live build's segmented row is no `v2` row, the Lean verifier's is: neither is canonical, both are bit rows
        assert!(!row_ok(&p(false, Some(vec![(20, 1)]))));
        assert!(bit_row(&p(false, Some(vec![(20, 1)]))) && bit_row(&p(true, None)) && !bit_row(&p(false, None)));
    }
}
