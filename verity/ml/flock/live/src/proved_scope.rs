//! The statement forms the soundness proof covers (verifier PROTOCOL.md §16.15): the Lean verifier's `Flock.ProvedScope.check`,
//! its rules in its order and its refusals word for word. The Lean verifier refuses every session of a statement outside
//! them, so a form no theorem covers is never accepted; the live server does the same ([`crate::StatementVerifier::refusal`]),
//! so its verdicts are the Lean verifier's.
//!
//! The live build writes only hm96-sha512 row statements with retained rounds and `hm96-sha512/v1` leaves, or, ZK off, plain
//! SHA-512 leaves (a circuit whose META names [`crate::circuit::leaf_scheme_plain`]; `ProvedScope.leafOk`), so the Lean
//! check's rules on those hold of every statement it sets up, as do its parse and its public file's load. Its statements
//! all have hidden outputs, so the circuit's check is `scopeOkI` once the wires passed: the general instance form. A
//! templated statement's is `scopeOkT` ([`tmpl_why`]), on its committed unit: the rows `flock-tmpl-unit` wrote, whose
//! `ORDER` line is the template's position map, one column per root column.

use crate::circuit::{port_cols, seg_shape, Composite, Instances, Port, STATEMENT, TMPL_STATEMENT};
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
    (n..net.const_pos).all(|r| row(&net.a, r).iter().chain(row(&net.b, r)).all(|&x| x >= n || mask[x]))
}

fn row(m: &[Vec<usize>], r: usize) -> &[usize] {
    m.get(r).map_or(&[], Vec::as_slice)
}

/// `ProvedScope.outReads`: for each output port of `net`, the input ports whose bits it reads, ascending; and whether its
/// computed rows and output ports read, among the input columns, only port bits, each row only earlier rows. A column's reads
/// are a mask, bit `p + 1` for input port `p` and bit `0` for an input column no port carries or a row read out of order; a
/// computed row's are the union of the columns' it reads, the constant reads nothing, and a column past it, which no row
/// computes, is read out of order. A lookup slot (`lookup`), which the verifier builds from its pinned table, is taken whole:
/// each output port reads every input port.
pub fn out_reads(net: &IrUnitNet, lookup: bool) -> (Vec<Vec<usize>>, bool) {
    let ins = port_cols(&net.in_groups);
    let outs = port_cols(&net.out_groups);
    if lookup {
        return (outs.iter().map(|_| (0..ins.len()).collect()).collect(), true);
    }
    let n = (net.in_words * WORD).min(net.const_pos);
    let words = (ins.len() + 1).div_ceil(64);
    let mut in_bit = vec![0usize; n];
    for (p, &(q, w)) in ins.iter().enumerate() {
        for x in q..(q + w).min(n) {
            in_bit[x] = p + 1;
        }
    }
    let n_rows = net.const_pos - n;
    let col = |rows: &[u64], done: usize, x: usize, m: &mut [u64]| {
        if x < n {
            m[in_bit[x] / 64] |= 1 << (in_bit[x] % 64);
        } else if x < net.const_pos {
            if x - n < done {
                m.iter_mut().zip(&rows[(x - n) * words..(x - n + 1) * words]).for_each(|(a, b)| *a |= b);
            } else {
                m[0] |= 1;
            }
        } else if x > net.const_pos {
            m[0] |= 1;
        }
    };
    let mut rows = vec![0u64; n_rows * words];
    let mut m = vec![0u64; words];
    for i in 0..n_rows {
        m.fill(0);
        for &x in row(&net.a, n + i).iter().chain(row(&net.b, n + i)) {
            col(&rows, i, x, &mut m);
        }
        rows[i * words..(i + 1) * words].copy_from_slice(&m);
    }
    let mut ok = rows.chunks(words).all(|r| r[0] & 1 == 0);
    let reads = outs.iter().map(|&(q, w)| {
        m.fill(0);
        for x in q..q + w {
            col(&rows, n_rows, x, &mut m);
        }
        ok &= m[0] & 1 == 0;
        (0..ins.len()).filter(|p| (m[(p + 1) / 64] >> ((p + 1) % 64)) & 1 == 1).collect()
    }).collect();
    (reads, ok)
}

/// **`ProvedScope.wiresAcyclic`**: outside the rows' `sha512x3` and `hm96` slots, starting from the units' leaf ports and the
/// leaf cuts, the wires fill every input port of every slot, each wire once every input port its source port reads in its own
/// slot ([`out_reads`]) is filled, in passes until one fills nothing (at most `wires.len() + 1`); and no row or output port of
/// a unit or stage reads an input column that is not a port bit.
pub fn wires_acyclic(c: &Composite) -> bool {
    let row_net = |n: usize| n == c.sha_net || n == c.hm_net;
    let n_ports: Vec<usize> = c.nets.iter().map(|e| port_cols(&e.1.in_groups).len()).collect();
    let slots = |n: usize| if row_net(n) { 0 } else { c.per_vu.get(n).copied().unwrap_or(0) };
    let mut base = vec![0usize];
    for n in 0..c.nets.len() {
        base.push(base[n] + slots(n) * n_ports[n]);
    }
    let node = |n: usize, i: usize, p: usize| base[n] + i * n_ports[n] + p;
    let in_slot = |n: usize, i: usize, p: usize| n < c.nets.len() && i < slots(n) && p < n_ports[n];
    let reads: Vec<(Vec<Vec<usize>>, bool)> = (0..c.nets.len())
        .map(|n| if slots(n) == 0 { (Vec::new(), true) } else { out_reads(&c.nets[n].1, c.lookups.contains_key(&n)) })
        .collect();
    let g0 = c.unit().in_groups.first().map_or(0, |g| g.bits.len());
    let mut av = vec![false; base[c.nets.len()]];
    for u in 0..c.upv {
        for p in 0..g0 {
            if in_slot(c.unit_net, u, p) {
                av[node(c.unit_net, u, p)] = true;
            }
        }
    }
    for &(u, j, _) in &c.leaf_cuts {
        if in_slot(c.unit_net, u, j) {
            av[node(c.unit_net, u, j)] = true;
        }
    }
    let ready = |av: &[bool], (sn, si, sp): (usize, usize, usize)| {
        sn < c.nets.len() && si < slots(sn) && reads[sn].0.get(sp).is_some_and(|ps| ps.iter().all(|&q| av[node(sn, si, q)]))
    };
    for _ in 0..=c.wires.len() {
        let mut filled = false;
        for w in &c.wires {
            let (dn, di, dp) = w.dst;
            if in_slot(dn, di, dp) && !av[node(dn, di, dp)] && ready(&av, w.src) {
                av[node(dn, di, dp)] = true;
                filled = true;
            }
        }
        if !filled {
            break;
        }
    }
    av.iter().all(|&x| x) && (0..c.nets.len()).all(|n| slots(n) == 0 || reads[n].1)
}

/// `ProvedScope.wiresFeed`: the unit reads input groups past its leaf group, or a tail stage returns the outputs.
pub fn wires_feed(c: &Composite) -> bool {
    c.unit().in_groups.len() != 1 || c.out_net != c.unit_net
}

/// `ProvedScope.wiresWhy`.
pub const WIRES_WHY: &str = "a slot no order of the wires computes from the leaves: a cycle through wires, an input port no wire \
                             fills, or a read of an input column that is not a port bit";

/// `ProvedScope.v1Ports`' row: a `sha512/row/v1` row of `16 · words` bits in at least one chunk.
fn v1_row(p: &Port) -> bool {
    !p.v2 && p.segs.is_none() && p.bits == 16 * p.words && p.comps > 0
}

/// `ProvedScope.v2Row`: an unsegmented `sha512/row/v2` row of `16 · words` bits, its bytes and SHA-512's 17 bytes of padding
/// filling its compressions' blocks and ending in its hm96 slot's block (the last compression's block may hold the bytes'
/// end and the padding's start).
fn v2_row(p: &Port) -> bool {
    p.v2 && p.segs.is_none() && p.bits == 16 * p.words && WORD * p.comps <= 2 * p.words + 16 &&
        2 * p.words + 17 <= WORD * (p.comps + 1)
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

/// `ProvedScope.scopeWhy hidden` of a circuit outside `scopeOkI` (`scopeOk` when not `hidden`), `None` inside: the unit has
/// at most `2^32` rows and reads only port bits among its inputs, and every row is a `sha512/row/v1` row, or, with hidden
/// outputs, its rows are [`bit_rows`]. Without hidden outputs the unit also reads only its leaf group and returns the
/// outputs; with them it may read any number of input groups and any stage may return the outputs, the general instance
/// form, which [`check`] takes once the wires are acyclic ([`wires_acyclic`]) wherever they feed the unit or the outputs.
pub fn circuit_why(c: &Composite, hidden: bool) -> Option<String> {
    let unit = c.unit();
    if !hidden && unit.in_groups.len() != 1 {
        Some(format!("a unit reading {} input groups", unit.in_groups.len()))
    } else if !hidden && c.out_net != c.unit_net {
        Some("a circuit whose outputs a tail stage computes (META out_net)".into())
    } else if unit.unit_log > 32 {
        Some(format!("a unit of 2^{} rows", unit.unit_log))
    } else if !port_reads(unit) {
        Some("a unit row reading an input column that is not a port bit".into())
    } else if let Some(why) = rows_why(&c.ports, &c.out_row_ports, hidden) {
        Some(why.into())
    } else if c.pinned > 0 {
        Some(PINNED_WHY.into())
    } else {
        None
    }
}

/// `ProvedScope.pinnedWhy`: a statement with pinned outputs, which no soundness theorem covers yet.
pub const PINNED_WHY: &str = "pinned outputs (META pinned_outputs): a form no soundness theorem covers yet";

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

/// `max 7 (log2ceil cols)`: the rows of a templated unit whose root has `cols` columns.
fn tmpl_unit_log(cols: usize) -> usize {
    crate::chunk::log2ceil(cols).max(7)
}

/// `ProvedScope.scopeWhyT` of a templated circuit outside `scopeOkT`, `None` inside: a `--zk` session of one table at hidden
/// outputs without the verifier's registered reads, whose unit reads only its leaf group and returns the outputs, of at most
/// `2^32` rows by its root's columns (the `ORDER` line's), and whose rows `scopeOkI` admits; past those, the refusal is
/// `scopeWhy true`'s ([`circuit_why`]'s). A templated statement with pinned outputs is refused as every other is.
pub fn tmpl_why(c: &Composite, hidden: bool, zk: bool, registered: bool, tables: usize) -> Option<String> {
    let unit = c.unit();
    let cols = unit.order.as_ref().map_or(0, Vec::len);
    if !hidden {
        Some("a templated statement without hidden outputs".into())
    } else if !zk {
        Some("a templated statement without --zk".into())
    } else if registered {
        Some("a templated statement with the verifier's registered reads (--registered)".into())
    } else if tables > 1 {
        Some(format!("a templated statement in {tables} tables"))
    } else if unit.in_groups.len() != 1 {
        Some(format!("a templated unit reading {} input groups", unit.in_groups.len()))
    } else if c.out_net != c.unit_net {
        Some("a templated circuit whose outputs a tail stage computes (META out_net)".into())
    } else if tmpl_unit_log(cols) > 32 {
        Some("a templated unit of more than 2^32 rows".into())
    } else if rows_why(&c.ports, &c.out_row_ports, true).is_some() {
        circuit_why(c, true)
    } else if c.pinned > 0 {
        Some(PINNED_WHY.into())
    } else {
        None
    }
}

/// `ProvedScope.zkTableCap`: the most tables a `--zk` session may have.
pub const ZK_TABLE_CAP: usize = 1 << 16;

/// `ProvedScope.statementRules`' first failure among those a live statement can fail: `name` is the statement as the Lean
/// verifier names it, `typed` whether it is `verity/flock-circuit/types`, `tables` the session's tables and `public_inputs`
/// whether it has public input ports.
pub fn statement_why(name: &str, typed: bool, zk: bool, tables: usize, public_inputs: bool) -> Option<String> {
    if typed {
        Some(format!("{name}, a typed statement"))
    } else if !zk && tables > 1 {
        Some(format!("{tables} tables in a session without --zk"))
    } else if zk && tables > ZK_TABLE_CAP {
        Some(format!("{tables} tables in a --zk session, more than {ZK_TABLE_CAP}"))
    } else if public_inputs {
        Some("public input ports (--public-inputs)".into())
    } else {
        None
    }
}

/// The statement's id as the Lean verifier names it: a selftest build's (`seed-injection`) is `+seed-injection`'s.
pub fn lean_name(c: &Composite) -> String {
    let base = match c.statement() {
        crate::typed::STATEMENT => crate::typed::STATEMENT,
        TMPL_STATEMENT => TMPL_STATEMENT,
        _ => STATEMENT,
    };
    format!("{base}{}", if cfg!(feature = "seed-injection") { "+seed-injection" } else { "" })
}

/// The shared-row rule of `ProvedScope.check` (`sharedRowsOk`): a shared-row public file (`shared`) is a covered form, with
/// or without `--zk`, only where the verifier holds its own registered reads (`registered`, `--registered F`) and the
/// statement's leaves are `hm96-sha512/v1` (not `plain_leaves`, a recursive session's inner statement). Every statement the
/// live build sets up has hidden outputs.
pub fn shared_rows_why(shared: bool, registered: bool, plain_leaves: bool) -> Option<&'static str> {
    (shared && !(registered && !plain_leaves)).then_some("a shared-row public file (header shared_rows)")
}

/// **`ProvedScope.check`** on a statement the live build set up: its circuit, its public file (`inst`), `--zk` and its tables.
/// The Lean verifier holds its own `--registered F` exactly when the public header names registered reads: without it such
/// a header is refused at setup, and with it a header naming none is (`Registered.check`).
pub fn check(c: &Composite, inst: &Instances, zk: bool, tables: usize) -> Result<(), String> {
    let typed = c.statement() == crate::typed::STATEMENT;
    if let Some(why) = statement_why(&lean_name(c), typed, zk, tables, !c.public_inputs.is_empty()) {
        return Err(refusal(&why));
    }
    let registered = inst.header.get("registered").is_some();
    if let Some(why) = shared_rows_why(inst.tables.is_some(), registered, c.plain_leaves()) {
        return Err(refusal(why));
    }
    if c.statement() == TMPL_STATEMENT {
        return tmpl_why(c, true, zk, registered, tables).map_or(Ok(()), |why| Err(refusal(&why)));
    }
    if wires_feed(c) && !wires_acyclic(c) {
        return Err(refusal(WIRES_WHY));
    }
    circuit_why(c, true).map_or(Ok(()), |why| Err(refusal(&why)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Wire;
    use crate::ir_block::{FlatCache, Group};
    use crate::lookup::LookupNet;

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
                    a, b, unit_log: 9, sha256: [0; 32], leaves: None, cut: None, order: None, flat: FlatCache::default() }
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
        assert_eq!(statement_why("verity/flock-circuit", false, true, ZK_TABLE_CAP + 1, true).unwrap(),
                   "65537 tables in a --zk session, more than 65536");
        assert_eq!(statement_why("verity/flock-circuit", false, true, ZK_TABLE_CAP, false), None);
    }

    fn port(words: usize, comps: usize, bits: usize, v2: bool, segs: Option<Vec<(usize, usize)>>) -> Port {
        Port { name: "x".into(), words, comps, bits, v2, segs }
    }

    #[test]
    fn rows_in_scope_are_lean_v1_or_v2_rows() {
        assert!(v1_row(&port(64, 1, 1024, false, None)));
        assert!(!v1_row(&port(64, 0, 1024, false, None)));
        assert!(!v1_row(&port(64, 1, 1024, true, None)));
        // a v2 row of 64 or 100 words ends, padded, in its slot's block
        assert!(v2_row(&port(64, 1, 1024, true, None)));
        assert!(v2_row(&port(100, 1, 1600, true, None)));
        // 32 words' 64 bytes and the padding fit the slot's block alone
        assert!(v2_row(&port(32, 0, 512, true, None)));
        // at 120 words its last 112 bytes leave no room for the padding: its second compression holds them and the
        // padding's start, its slot's block the rest
        assert!(!v2_row(&port(120, 1, 1920, true, None)));
        assert!(v2_row(&port(120, 2, 1920, true, None)));
        assert!(!v2_row(&port(120, 3, 1920, true, None)));
        assert!(!v2_row(&port(32, 1, 512, true, None)));
        assert!(!v2_row(&port(64, 1, 1000, true, None)));
        assert!(!v2_row(&port(64, 1, 1024, true, Some(vec![(1024, 1)]))));
        assert!(!v2_row(&port(64, 1, 1024, false, None)));
    }

    #[test]
    fn v2_rows_are_the_parsers() {
        // Lean's Flock/Test/SpillRows.lean, case for case: a row of w leaves is in scope at the compressions the parser
        // derives from its length, input or output row, and at one more or one fewer not
        let serial = |w: usize| (2 * w + 17 + 127) / 128 - 1;
        for w in 0..=2048 {
            assert!(v2_row(&port(w, serial(w), 16 * w, true, None)), "{w}");
            assert!(!v2_row(&port(w, serial(w) + 1, 16 * w, true, None)), "{w}");
            assert!(serial(w) == 0 || !v2_row(&port(w, serial(w) - 1, 16 * w, true, None)), "{w}");
        }
        assert!(serial(56) == 1 && v2_row(&port(56, 1, 896, true, None)) && !v2_row(&port(56, 0, 896, true, None)));
        let row = |w: usize, n: usize| port(w, n, 16 * w, true, None);
        assert!(bit_rows(&[row(56, 1), row(1, 0)], &[1]));
        assert!(bit_rows(&[row(1, 0), row(56, 1)], &[1]));
        assert!(bit_rows(&[row(120, 2), row(248, 4)], &[1]));
        assert!(!bit_rows(&[row(1, 0), row(56, 2)], &[1]));
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
        // a row's padding may spill into its slot's block, an input row's or an output row's
        let spill = port(120, 2, 1920, true, None);
        assert_eq!(rows_why(&[spill.clone(), port(1, 0, 16, true, None)], &[1], true), None);
        assert_eq!(rows_why(&[port(1, 0, 16, true, None), spill], &[1], true), None);
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
    fn a_shared_row_file_needs_registered_reads_and_hm96_leaves() {
        let refused = Some("a shared-row public file (header shared_rows)");
        assert_eq!(shared_rows_why(true, true, false), None);
        assert_eq!(shared_rows_why(true, false, false), refused);
        assert_eq!(shared_rows_why(true, true, true), refused);
        assert_eq!(shared_rows_why(true, false, true), refused);
        assert_eq!(shared_rows_why(false, false, false), None);
        assert_eq!(shared_rows_why(false, false, true), None);
        assert_eq!(refusal(refused.unwrap()),
                   "refused: a shared-row public file (header shared_rows): a statement form the soundness proof does not cover");
    }

    /// Lean's `Flock.Test.Wires.net`: `in_words` input words with port groups `groups`, then `rows`, then the constant; its
    /// output ports, of widths `outs`, start at the first computed row.
    fn wnet(groups: Vec<Group>, in_words: usize, outs: Vec<usize>, rows: Vec<(Vec<usize>, Vec<usize>)>) -> IrUnitNet {
        let mut n = net(groups, in_words, rows);
        n.out_groups = vec![Group { col: in_words, words: 1, bits: outs }];
        n
    }

    /// Lean's `Flock.Test.Wires.circ`: the unit first (`per_vu[0]` of them), then the rows' two nets after `nets`.
    fn circ(mut nets: Vec<IrUnitNet>, mut per_vu: Vec<usize>, wires: &[(usize, usize, usize, usize, usize, usize)],
            leaf_cuts: Vec<(usize, usize, usize)>, lookups: Vec<(usize, LookupNet)>) -> Composite {
        let k = nets.len();
        nets.extend([wnet(vec![], 0, vec![], vec![]), wnet(vec![], 0, vec![], vec![])]);
        per_vu.extend([1, 1]);
        Composite {
            name: "wires".into(), sha512: [0; 64], class_sha512: [0; 64], meta: serde_json::Value::Null,
            nets: nets.into_iter().map(|n| ("net".to_string(), n)).collect(), ranges: vec![], ports: vec![], n_in: 0,
            out_words: vec![], out_row_ports: vec![], pinned: 0, g: 1, k_log: 0, upv: per_vu[0], unit_net: 0, leaves_in: vec![],
            leaves_out: vec![], public_inputs: vec![], leaves_public: vec![], rows: vec![], sha_net: k, hm_net: k + 1, out_net: 0,
            out_ports: vec![], out_slotted: false, leaf_cuts_of: vec![], leaf_cuts, dummy_bc: vec![], dummy_outputs: vec![],
            leaf_scheme: serde_json::Value::Null, per_vu,
            wires: wires.iter().map(|&(a, b, c, d, e, f)| Wire { dst: (a, b, c), src: (d, e, f) }).collect(),
            lookups: lookups.into_iter().collect(), typed: None,
        }
    }

    /// Lean's `unit`: leaf port `x` (column 0), cut port `s` (column 128); it returns `x·x` (port 0) and `x·s` (port 1).
    fn unit() -> IrUnitNet {
        let g = |col| Group { col, words: 1, bits: vec![1] };
        wnet(vec![g(0), g(1)], 2, vec![1, 1], vec![(vec![0], vec![0]), (vec![0], vec![128])])
    }

    /// Lean's `stage k`: `k` one-bit input ports, returning their XOR (port 0) and the constant (port 1, column 130).
    fn stage(k: usize) -> IrUnitNet {
        wnet(vec![Group { col: 0, words: 1, bits: vec![1; k] }], 1, vec![1, 1], vec![((0..k).collect(), vec![130]), (vec![130], vec![130])])
    }

    fn rms(w0: usize) -> Composite {
        circ(vec![unit(), stage(2)], vec![2, 1], &[(1, 0, 0, 0, 0, w0), (1, 0, 1, 0, 1, w0), (0, 0, 1, 1, 0, 0), (0, 1, 1, 1, 0, 0)],
             vec![], vec![])
    }

    fn gemm(n: usize) -> Vec<(usize, usize, usize, usize, usize, usize)> {
        let mut w = vec![(0, 0, 1, 1, 0, 1)];
        w.extend((0..n - 1).map(|i| (0, i + 1, 1, 0, i, 1)));
        w.push((1, 0, 0, 0, n - 1, 1));
        w
    }

    #[test]
    fn wires_are_lean_acyclic() {
        // Lean's Flock/Test/Wires.lean, case for case
        assert_eq!(out_reads(&unit(), false), (vec![vec![0], vec![0, 1]], true));
        assert_eq!(out_reads(&stage(2), false), (vec![vec![0, 1], vec![]], true));
        assert!(wires_acyclic(&rms(0)));
        assert!(!wires_acyclic(&rms(1)));
        let g = |w: &[_]| wires_acyclic(&circ(vec![unit(), stage(1)], vec![4, 1], w, vec![], vec![]));
        let mut w = gemm(4);
        assert!(g(&w));
        w.reverse();
        assert!(g(&w));
        w = gemm(4);
        w[0] = (0, 0, 1, 1, 0, 0);
        assert!(!g(&w));
        assert!(!wires_acyclic(&circ(vec![unit()], vec![1], &[], vec![], vec![])));
        assert!(wires_acyclic(&circ(vec![unit()], vec![1], &[], vec![(0, 1, 7)], vec![])));
        assert!(!wires_acyclic(&circ(vec![unit(), stage(1)], vec![1, 1], &[(0, 0, 1, 1, 1, 1), (1, 0, 0, 0, 0, 0)], vec![], vec![])));
        let one = || vec![Group { col: 0, words: 1, bits: vec![1] }];
        let mut loose_out = wnet(one(), 1, vec![1], vec![(vec![0], vec![0])]);
        loose_out.out_groups = vec![Group { col: 0, words: 1, bits: vec![1, 1] }];
        for loose in [wnet(one(), 1, vec![1], vec![(vec![0], vec![5])]), loose_out] {
            assert!(!out_reads(&loose, false).1);
            assert!(!wires_acyclic(&circ(vec![unit(), loose.clone()], vec![1, 1], &[(0, 0, 1, 1, 0, 0), (1, 0, 0, 0, 0, 0)], vec![], vec![])));
            assert!(wires_acyclic(&circ(vec![unit(), loose], vec![1, 0], &[], vec![(0, 1, 0)], vec![])));
        }
        // Lean's `past k`: output ports on the first `k` columns from 128, the constant at 129
        let past = |k: usize| wnet(one(), 1, vec![1; k], vec![(vec![0], vec![0])]);
        assert_eq!(out_reads(&past(2), false), (vec![vec![0], vec![]], true));
        assert!(!out_reads(&past(3), false).1);
        assert!(!wires_acyclic(&circ(vec![unit(), past(3)], vec![1, 1], &[(0, 0, 1, 1, 0, 2), (1, 0, 0, 0, 0, 0)], vec![], vec![])));
        // a lookup slot between two stages, taken whole: Lean's is a one-row stand-in, this one a 16-word table's
        let table: &'static [u32] = Box::leak(vec![0u32; 16].into_boxed_slice());
        let lookup = |back: bool| {
            let (lnet, fast) = LookupNet::build("lookup:t", table, 4, 2, 31);
            circ(vec![unit(), stage(1), lnet, stage(1)], vec![1, 1, 1, 1],
                 &[(1, 0, 0, 0, 0, 0), (2, 0, 0, if back { 3 } else { 1 }, 0, 0), (3, 0, 0, 2, 0, 0), (0, 0, 1, 3, 0, 0)], vec![], vec![(2, fast)])
        };
        assert!(wires_acyclic(&lookup(false)));
        assert!(!wires_acyclic(&lookup(true)));
        assert!(wires_feed(&rms(1)));
        assert!(!wires_feed(&circ(vec![wnet(one(), 1, vec![1], vec![(vec![0], vec![0])])], vec![1], &[], vec![], vec![])));
        assert_eq!(refusal(WIRES_WHY), "refused: a slot no order of the wires computes from the leaves: a cycle through wires, an input \
                                        port no wire fills, or a read of an input column that is not a port bit: a statement form the \
                                        soundness proof does not cover");
    }

    /// Lean's `tail`: a unit of one group and a stage it feeds, the stage returning the outputs.
    fn tail() -> Composite {
        let mut c = circ(vec![wnet(vec![Group { col: 0, words: 1, bits: vec![1] }], 1, vec![1], vec![(vec![0], vec![0])]), stage(1)],
                         vec![1, 1], &[(1, 0, 0, 0, 0, 0)], vec![], vec![]);
        c.out_net = 1;
        c
    }

    #[test]
    fn hidden_outputs_take_the_general_instance_form() {
        // Lean's Flock/Test/Wires.lean: a unit reading a cut group, or a stage returning the outputs, once the wires are
        // acyclic; with public outputs, the unit's form alone
        assert!(wires_acyclic(&tail()) && wires_feed(&tail()));
        assert_eq!(circuit_why(&rms(0), true), None);
        assert_eq!(circuit_why(&tail(), true), None);
        assert_eq!(circuit_why(&rms(0), false).as_deref(), Some("a unit reading 2 input groups"));
        assert_eq!(circuit_why(&tail(), false).as_deref(), Some("a circuit whose outputs a tail stage computes (META out_net)"));
        assert!(wires_acyclic(&slotted()) && wires_feed(&slotted()));
        assert_eq!(circuit_why(&slotted(), true), None);
    }

    /// Lean's `slotted`: a repeated last stage, two units each wiring its own slot of a stage of two 16-bit output ports,
    /// the stage's slots returning the output words (`out_net_ports` of `[slot, port]` pairs).
    fn slotted() -> Composite {
        let one = || vec![Group { col: 0, words: 1, bits: vec![1] }];
        let stage = wnet(one(), 1, vec![16, 16], vec![(vec![0], vec![0]); 32]);
        let mut c = circ(vec![wnet(one(), 1, vec![1], vec![(vec![0], vec![0])]), stage], vec![2, 2],
                         &[(1, 0, 0, 0, 0, 0), (1, 1, 0, 0, 1, 0)], vec![], vec![]);
        c.out_net = 1;
        c.out_ports = vec![(0, 0), (1, 0)];
        c.out_slotted = true;
        c
    }

    /// `c` as a templated statement's circuit, its unit committed at a position map of `cols` columns.
    fn templated(mut c: Composite, cols: usize) -> Composite {
        c.meta = serde_json::json!({ "statement": TMPL_STATEMENT });
        c.nets[c.unit_net].1.order = Some((0..cols).collect());
        c
    }

    fn flat_unit() -> Composite {
        circ(vec![wnet(vec![Group { col: 0, words: 1, bits: vec![1] }], 1, vec![1], vec![(vec![0], vec![0])])], vec![1], &[], vec![], vec![])
    }

    #[test]
    fn a_templated_statement_takes_lean_scope_ok_t() {
        // Lean's scopeWhyT, rule for rule and in its order
        let c = templated(flat_unit(), 130);
        assert_eq!(tmpl_why(&c, true, true, false, 1), None);
        let why = |hidden, zk, registered, tables| tmpl_why(&c, hidden, zk, registered, tables).unwrap();
        assert_eq!(why(false, false, true, 2), "a templated statement without hidden outputs");
        assert_eq!(why(true, false, true, 2), "a templated statement without --zk");
        assert_eq!(why(true, true, true, 2), "a templated statement with the verifier's registered reads (--registered)");
        assert_eq!(why(true, true, false, 2), "a templated statement in 2 tables");
        assert_eq!(refusal(&why(true, false, false, 1)),
                   "refused: a templated statement without --zk: a statement form the soundness proof does not cover");
        let two = templated(rms(0), 130);
        assert_eq!(tmpl_why(&two, true, true, false, 1).as_deref(), Some("a templated unit reading 2 input groups"));
        let tail = templated(tail(), 130);
        assert_eq!(tmpl_why(&tail, true, true, false, 1).as_deref(),
                   Some("a templated circuit whose outputs a tail stage computes (META out_net)"));
        assert_eq!((tmpl_unit_log(0), tmpl_unit_log(130), tmpl_unit_log(1 << 32), tmpl_unit_log((1 << 32) + 1)), (7, 8, 32, 33));
    }

    #[test]
    fn a_templated_statement_past_its_description_is_refused_as_scope_why_true() {
        // the rows scopeOkI admits pass whatever the unit's own size; past them, scopeWhy true's first failure
        let mut c = templated(flat_unit(), 130);
        c.nets[0].1.unit_log = 33;
        assert_eq!(tmpl_why(&c, true, true, false, 1), None);
        c.ports = vec![port(64, 1, 1000, true, None)];
        assert_eq!(tmpl_why(&c, true, true, false, 1).as_deref(), Some("a unit of 2^33 rows"));
        c.nets[0].1.unit_log = 9;
        assert!(tmpl_why(&c, true, true, false, 1).unwrap().starts_with("rows that are neither"));
        c.ports = vec![];
        c.pinned = 1;
        assert_eq!(tmpl_why(&c, true, true, false, 1).as_deref(), Some(PINNED_WHY));
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
