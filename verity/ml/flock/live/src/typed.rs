//! `verity/flock-circuit/types`: the circuit statement whose unit is circuit types under layouts, never rows
//! (`verity_flock.typed_statement`; Project doc `constant-api-public.md` §2.2). The verifier derives the unit's rows from
//! them (the Lean `Flock.Derive`). This module is the Rust prover's mirror of that derivation, as `verity_flock.derive` is
//! Python's.
//!
//! **A flat class:** the unit is one type of AND items, under a layout that places nothing. [`unit_rows`] writes its rows
//! as the `flock-ir-unit/v2` text the flat statement's unit section holds, so the rest of the statement reads as the flat
//! statement's.
//!
//! **A template with a tail** (GEMM): the unit is its instance type, which calls other types, each placed by the block in
//! a range of its layout's own ([`Instance`]). Rows sit in unit coordinates: the unit's own region first, then each part
//! (placed call) aligned to its range, in item order.
//! - A layout's own region follows the v2 rule: the input port groups (aligned power-of-two runs of 128-bit words, exported
//!   bits left out), one row per AND item, the output port groups' copy rows, and the constant last.
//! - A placed call's input bit is exported when its form is exactly one input wire of the caller that nothing else reads:
//!   that wire's column is the part's input row.
//! - Every other input row of a part is bound to its form over the unit's columns ([`Instance::bind`]), the block's Δ.
//! - The unit's own rows read its parts' output columns directly: in the block those entries cross slots.
//! - A table read (attention's MUFU reads) is a part too, placed in the range of a generated layout: its rows are
//!   `table/v2`'s (`LookupNet::build_v2`, the verifier's `Lookup.buildV2`) over the held table, its index inputs are always
//!   bound, and its product rows' B side is the table's, folded from it ([`TableSide`]) and listed empty.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};
use sha2::{Digest, Sha512};

use crate::{ir_block::IrUnitNet, lookup::LookupNet};

pub const FORMAT: &str = "flock-circuit/types";
pub const STATEMENT: &str = "verity/flock-circuit/types";
const POLICY: &str = "calls/v1";
const TYPE_FORMAT: &str = "verity/circuit-type/v1";
const LAYOUT_FORMAT: &str = "verity/layout/v1";
const WORD: usize = 128;
/// the constant's column, until the own region's last row is known
const KONST: i64 = -1;

/// `[[w, ...], c]`: the XOR of ascending, distinct wires (or columns) and the constant `c`
#[derive(Clone, Debug, PartialEq)]
struct Form {
    wires: Vec<usize>,
    c: bool,
}

#[derive(Debug)]
enum Item {
    And(Form, Form),
    /// `calls[k]` on its input forms
    Call(usize, Vec<Form>),
    /// `tables[t]` at its index forms
    Read(usize, Vec<Form>),
}

/// A `verity/circuit-type/v1` type of AND items, calls and table reads: its ports, items and outputs.
#[derive(Debug)]
pub struct Type {
    pub ins: Vec<(usize, usize)>,
    pub outs: Vec<(usize, usize)>,
    calls: Vec<String>,
    /// `(SHA-512 of its raw words, index bits, value bits)` per table
    tables: Vec<(String, usize, usize)>,
    items: Vec<Item>,
    outputs: Vec<Form>,
}

/// A layout's own region: derived from its type, or a `table/v2` read over a held table.
#[derive(Clone, Debug, PartialEq)]
pub enum Own {
    Derive,
    Gen { table: String, in_bits: usize, value_bits: usize, lo_bits: usize },
}

/// A `verity/layout/v1` record: its type (none for a generated read), range, port groups and placed calls and reads.
#[derive(Debug)]
pub struct Layout {
    pub sha512: String,
    pub ty: Option<String>,
    /// absent for the unit's layout, which the block places
    pub range_log: Option<usize>,
    pub in_split: Vec<usize>,
    pub out_split: Vec<usize>,
    pub own: Own,
    /// `(item, layout, offset)` per call item, in item order
    calls: Vec<(usize, usize, usize)>,
    /// `(item, layout, offset)` per read item, in item order
    reads: Vec<(usize, usize, usize)>,
}

/// A layout's own region, the constant resolved.
#[derive(Clone, Debug)]
pub struct Rows {
    pub rows: Vec<(Vec<usize>, Vec<usize>)>,
    pub konst: usize,
    /// per group: first word, words, each port's bits
    pub in_groups: Vec<(usize, usize, Vec<usize>)>,
    pub out_groups: Vec<(usize, usize, Vec<usize>)>,
    /// each input bit's column (a part's input row for an exported bit) and each output bit's copy row
    pub in_cols: Vec<usize>,
    pub out_cols: Vec<usize>,
    /// a generated read's `READ` record (its product rows' B side, listed empty): table SHA-512, n, lo, k, prod0, lo_top
    pub read: Option<(String, usize, usize, usize, usize, usize)>,
}

fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), sorted(x))).collect::<BTreeMap<_, _>>().into_iter().collect()),
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        x => x.clone(),
    }
}

/// `SHA-512(domain || 0 || canon(v))`, canon RFC 8785 JSON: for a statement's values (integers, ASCII strings, arrays,
/// objects), keys sorted and no whitespace.
pub fn digest(domain: &str, v: &Value) -> String {
    let mut h = Sha512::new();
    h.update(domain.as_bytes());
    h.update([0u8]);
    h.update(serde_json::to_string(&sorted(v)).unwrap().as_bytes());
    crate::hex(&h.finalize())
}

/// A type content's digest.
pub fn type_digest(content: &Value) -> String {
    digest(TYPE_FORMAT, content)
}

/// A layout record's `sha512`: the digest of the record without it.
pub fn layout_digest(rec: &Value) -> String {
    let mut r = rec.clone();
    if let Some(m) = r.as_object_mut() {
        m.remove("sha512");
    }
    digest(LAYOUT_FORMAT, &r)
}

fn uint(v: &Value, what: &str) -> Result<usize, String> {
    v.as_u64().map(|x| x as usize).ok_or(format!("{what} is not a non-negative integer"))
}

fn form(v: &Value, n: usize) -> Result<Form, String> {
    let a = v.as_array().filter(|a| a.len() == 2).ok_or("a form is not [[wires], c]")?;
    let wires = a[0].as_array().ok_or("a form's wires")?.iter().map(|w| uint(w, "a form's wire")).collect::<Result<Vec<_>, _>>()?;
    if wires.windows(2).any(|p| p[0] >= p[1]) || wires.last().is_some_and(|&w| w >= n) {
        return Err("a form's wires are not ascending, distinct and defined before it".into());
    }
    let c = match a[1].as_u64() {
        Some(0) => false,
        Some(1) => true,
        _ => return Err("a form's constant is neither 0 nor 1".into()),
    };
    Ok(Form { wires, c })
}

fn ports(v: &Value) -> Result<Vec<(usize, usize)>, String> {
    v.as_array().ok_or("a type's ports")?.iter().map(|p| match p.as_array().map(|a| a.as_slice()) {
        Some([n, w]) if w.as_u64().is_some_and(|w| w >= 1) => Ok((uint(n, "a port's count")?, uint(w, "a port's width")?)),
        _ => Err("a port is not [count, width >= 1]".to_string()),
    }).collect()
}

fn bits(ports: &[(usize, usize)]) -> Vec<usize> {
    ports.iter().map(|&(n, w)| n * w).collect()
}

impl Type {
    pub fn in_bits(&self) -> usize {
        bits(&self.ins).iter().sum()
    }

    pub fn out_bits(&self) -> usize {
        bits(&self.outs).iter().sum()
    }

    /// The type `content` encodes: AND items, calls and table reads. `callee(digest)` is a called type's
    /// `(in bits, out bits)`.
    pub fn parse(content: &Value, callee: &dyn Fn(&str) -> Option<(usize, usize)>) -> Result<Self, String> {
        let keys = ["format", "label", "in", "out", "calls", "tables", "items", "outputs"];
        if content["format"] != TYPE_FORMAT || content.as_object().is_none_or(|m| m.len() != keys.len() || keys.iter().any(|k| !m.contains_key(*k))) {
            return Err(format!("a type is not a {TYPE_FORMAT} content"));
        }
        if !["library", "program"].iter().any(|l| content["label"] == *l) {
            return Err("a type's label is neither library nor program".into());
        }
        let tables: Vec<(String, usize, usize)> = content["tables"].as_array().ok_or("a type's tables")?.iter().map(|t| match t.as_array().map(|a| a.as_slice()) {
            Some([sha, ib, vb]) => {
                let sha = sha.as_str().filter(|x| x.len() == 128 && x.bytes().all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c)))
                    .ok_or("a table is not named by its SHA-512")?;
                let (ib, vb) = (uint(ib, "a table's index bits")?, uint(vb, "a table's value bits")?);
                if !(1..=32).contains(&ib) || !(1..=32).contains(&vb) {
                    return Err("a table's index or value bits are not in 1..32".to_string());
                }
                Ok((sha.to_string(), ib, vb))
            }
            _ => Err("a table is not [sha512, index_bits, value_bits]".to_string()),
        }).collect::<Result<_, _>>()?;
        if tables.iter().map(|t| &t.0).collect::<BTreeSet<_>>().len() != tables.len() {
            return Err("a type lists a table twice".into());
        }
        let calls: Vec<String> = content["calls"].as_array().ok_or("a type's calls")?.iter()
            .map(|d| d.as_str().map(str::to_string).ok_or("a call is not a type digest".to_string())).collect::<Result<_, _>>()?;
        let shapes = calls.iter().map(|d| callee(d).ok_or(format!("a call of type {}.., not among the statement's", d.get(..16).unwrap_or(d))))
            .collect::<Result<Vec<_>, _>>()?;
        let (ins, outs) = (ports(&content["in"])?, ports(&content["out"])?);
        let mut n: usize = bits(&ins).iter().sum();
        let mut items = Vec::new();
        for it in content["items"].as_array().ok_or("a type's items")? {
            match it.as_array().map(|a| a.as_slice()) {
                Some([op, a, b]) if op.as_u64() == Some(0) => {
                    items.push(Item::And(form(a, n)?, form(b, n)?));
                    n += 1;
                }
                Some([op, k, args]) if op.as_u64() == Some(3) => {
                    let k = uint(k, "a call's type index")?;
                    let (ib, ob) = *shapes.get(k).ok_or("a call of a type its calls do not list")?;
                    let args = args.as_array().ok_or("a call's arguments")?.iter().map(|f| form(f, n)).collect::<Result<Vec<_>, _>>()?;
                    if args.len() != ib {
                        return Err("a call does not pass its callee's input bits".into());
                    }
                    items.push(Item::Call(k, args));
                    n += ob;
                }
                Some([op, t, args]) if op.as_u64() == Some(4) => {
                    let t = uint(t, "a read's table index")?;
                    let (ib, vb) = tables.get(t).map(|x| (x.1, x.2)).ok_or("a read of a table its tables do not list")?;
                    let args = args.as_array().ok_or("a read's index")?.iter().map(|f| form(f, n)).collect::<Result<Vec<_>, _>>()?;
                    if args.len() != ib {
                        return Err("a read's index is not its table's index bits".into());
                    }
                    items.push(Item::Read(t, args));
                    n += vb;
                }
                _ => return Err("an item that is neither an AND, a call nor a read".into()),
            }
        }
        let outputs = content["outputs"].as_array().ok_or("a type's outputs")?.iter().map(|f| form(f, n)).collect::<Result<Vec<_>, _>>()?;
        if outputs.len() != bits(&outs).iter().sum::<usize>() {
            return Err("a type's outputs are not its out ports' bits".into());
        }
        Ok(Type { ins, outs, calls, tables, items, outputs })
    }
}

/// Every type of `META`, held to its digest; callees parsed before their callers.
fn types(meta: &Value) -> Result<BTreeMap<String, Type>, String> {
    let all = meta["types"].as_object().ok_or("META: types")?;
    fn visit(d: &str, all: &serde_json::Map<String, Value>, done: &mut BTreeMap<String, Type>, on: &mut BTreeSet<String>) -> Result<(), String> {
        if done.contains_key(d) {
            return Ok(());
        }
        let content = all.get(d).ok_or(format!("META: type {}.. is not among its types", d.get(..16).unwrap_or(d)))?;
        if type_digest(content) != d {
            return Err(format!("META: type {}..'s digest is not its content's", d.get(..16).unwrap_or(d)));
        }
        if !on.insert(d.to_string()) {
            return Err("META: a type calls itself".into());
        }
        for c in content["calls"].as_array().map(|a| a.as_slice()).unwrap_or(&[]) {
            visit(c.as_str().unwrap_or(""), all, done, on)?;
        }
        let t = Type::parse(content, &|c| done.get(c).map(|u| (u.in_bits(), u.out_bits())))?;
        done.insert(d.to_string(), t);
        Ok(())
    }
    let (mut done, mut on) = (BTreeMap::new(), BTreeSet::new());
    for d in all.keys() {
        visit(d, all, &mut done, &mut on)?;
    }
    Ok(done)
}

impl Layout {
    /// The layout `rec` records: a derived own region (of its type) or a generated `table/v2` read (no type, in a range,
    /// placing nothing); every call and read placed. Its `sha512` must be the record's digest.
    pub fn parse(rec: &Value) -> Result<Self, String> {
        let m = rec.as_object().ok_or("a layout is not an object")?;
        if rec["sha512"].as_str() != Some(layout_digest(rec).as_str()) {
            return Err("a layout's sha512 is not its record's digest".into());
        }
        let keys = ["sha512", "in_split", "out_split", "own", "calls", "reads"];
        if keys.iter().any(|k| !m.contains_key(*k)) || m.keys().any(|k| !keys.contains(&k.as_str()) && k != "type" && k != "range_log") {
            return Err("a layout record's keys are not verity/layout/v1's".into());
        }
        let own = if rec["own"] == json!({"derive": "v2"}) {
            Own::Derive
        } else {
            let o = rec["own"].as_object().ok_or("a layout's own")?;
            let gen_keys: BTreeSet<&str> = o.keys().map(String::as_str).collect();
            if rec["own"]["gen"] != "table/v2" || gen_keys != BTreeSet::from(["gen", "table", "in_bits", "value_bits", "lo_bits"]) {
                return Err("a layout's own region is neither derive v2 nor gen table/v2".into());
            }
            Own::Gen {
                table: rec["own"]["table"].as_str().ok_or("a generated read's table")?.to_string(), in_bits: uint(&rec["own"]["in_bits"], "in_bits")?,
                value_bits: uint(&rec["own"]["value_bits"], "value_bits")?, lo_bits: uint(&rec["own"]["lo_bits"], "lo_bits")?,
            }
        };
        let range_log = match rec.get("range_log") {
            None => None,
            Some(s) => Some(uint(s, "a layout's range_log").ok().filter(|s| (7..=40).contains(s)).ok_or("a layout's range_log is not in 7..40")?),
        };
        let placed = |k: &str| rec[k].as_array().ok_or(format!("a layout's {k}"))?.iter().map(|e| {
            let keys: BTreeSet<&str> = e.as_object().map(|m| m.keys().map(String::as_str).collect()).unwrap_or_default();
            if keys != BTreeSet::from(["item", "layout", "offset"]) {
                return Err(format!("a {k} entry that is not placed (inline): not read yet"));
            }
            Ok((uint(&e["item"], "an entry's item")?, uint(&e["layout"], "an entry's layout")?, uint(&e["offset"], "an entry's offset")?))
        }).collect::<Result<Vec<_>, _>>();
        let split = |k: &str| rec[k].as_array().ok_or(format!("a layout's {k}"))?.iter().map(|x| uint(x, k)).collect::<Result<Vec<_>, _>>();
        let lay = Layout {
            sha512: rec["sha512"].as_str().unwrap_or("").to_string(), ty: rec.get("type").map(|t| t.as_str().map(str::to_string).ok_or("a layout's type")).transpose()?,
            range_log, in_split: split("in_split")?, out_split: split("out_split")?, own, calls: placed("calls")?, reads: placed("reads")?,
        };
        let generated = lay.own != Own::Derive;
        if generated == lay.ty.is_some() || (generated && (lay.range_log.is_none() || !lay.calls.is_empty() || !lay.reads.is_empty()
            || !lay.in_split.is_empty() || !lay.out_split.is_empty()))
        {
            return Err("a layout is not a type's derived region, nor a generated read in a range placing nothing".into());
        }
        Ok(lay)
    }
}

fn ceil_pow2(n: usize) -> usize {
    n.max(1).next_power_of_two()
}

/// Each port group's (first bit, bits, first port, end port) over ports split at `split`.
fn groups(bits: &[usize], split: &[usize]) -> Result<Vec<(usize, usize, usize, usize)>, String> {
    if split.windows(2).any(|p| p[0] >= p[1]) || split.iter().any(|&s| s == 0 || s >= bits.len()) {
        return Err("a layout's split is not ascending port indices inside the ports".into());
    }
    let cuts: Vec<usize> = std::iter::once(0).chain(split.iter().copied()).chain(std::iter::once(bits.len())).collect();
    let first = |p: usize| bits[..p].iter().sum::<usize>();
    Ok(cuts.windows(2).filter(|c| c[1] > c[0]).map(|c| (first(c[0]), first(c[1]) - first(c[0]), c[0], c[1])).collect())
}

/// The forms of `f`'s wires, XORed: a form over columns.
fn xor(wire: &[Form], f: &Form) -> Form {
    let mut cols = BTreeSet::new();
    let mut c = f.c;
    for &w in &f.wires {
        for &x in &wire[w].wires {
            if !cols.remove(&x) {
                cols.insert(x);
            }
        }
        c ^= wire[w].c;
    }
    Form { wires: cols.into_iter().collect(), c }
}

fn entries(f: &Form) -> Vec<i64> {
    f.wires.iter().map(|&x| x as i64).chain(f.c.then_some(KONST)).collect()
}

/// A row of the unit's evaluation order: an own row, or a part (its bound inputs, then its rows).
#[derive(Clone, Copy, Debug)]
enum Step {
    Row(usize),
    Part(usize),
}

/// A layout's own region under `lay` (`verity_flock.derive.Derivation`, the v2 rule), its placed calls at `subs`
/// (`(base, callee rows)` per call item, in item order). Returns the rows, the evaluation order of its items, and each
/// call's bound input rows with their sources (the placed calls' non-exported inputs), in call order.
#[allow(clippy::type_complexity)]
fn own_region(t: &Type, lay: &Layout, subs: &[(usize, &Rows)]) -> Result<(Rows, Vec<Step>, Vec<Vec<(usize, Vec<usize>)>>), String> {
    let in_bits = t.in_bits();
    // a wire's reads: every item form's and every output's
    let mut uses: BTreeMap<usize, usize> = BTreeMap::new();
    let forms = t.items.iter().flat_map(|it| match it { Item::And(a, b) => vec![a, b], Item::Call(_, args) | Item::Read(_, args) => args.iter().collect() });
    for f in forms.chain(&t.outputs) {
        for &w in &f.wires {
            *uses.entry(w).or_default() += 1;
        }
    }
    // placed calls and reads, in item order, take `subs` in turn; only a call's arguments are ever exported
    let placed: Vec<(bool, &Vec<Form>)> = t.items.iter().filter_map(|it| match it {
        Item::Call(_, a) => Some((true, a)),
        Item::Read(_, a) => Some((false, a)),
        Item::And(..) => None,
    }).collect();
    if placed.len() != subs.len() {
        return Err("a layout does not place every call and read of its type".into());
    }
    let mut export_col: BTreeMap<usize, usize> = BTreeMap::new();
    let mut exported: Vec<BTreeSet<usize>> = Vec::new();
    for (&(call, args), &(base, sub)) in placed.iter().zip(subs) {
        let mut ex = BTreeSet::new();
        for (bit, f) in args.iter().enumerate() {
            if call && !f.c && f.wires.len() == 1 && f.wires[0] < in_bits && uses.get(&f.wires[0]) == Some(&1) {
                ex.insert(bit);
                export_col.insert(f.wires[0], base + sub.in_cols[bit]);
            }
        }
        exported.push(ex);
    }
    let mut rows: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
    let pad = |rows: &mut Vec<(Vec<i64>, Vec<i64>)>, to: usize| rows.resize(to, (Vec::new(), Vec::new()));
    let size_of = |bits: usize| WORD * ceil_pow2(bits.div_ceil(WORD));
    let ib = bits(&t.ins);
    let mut wire: Vec<Form> = vec![Form { wires: Vec::new(), c: false }; in_bits];
    let mut in_cols = vec![0usize; in_bits];
    let mut in_groups = Vec::new();
    for (first, count, pa, pb) in groups(&ib, &lay.in_split)? {
        let kept: Vec<usize> = (first..first + count).filter(|w| !export_col.contains_key(w)).collect();
        let size = size_of(kept.len());
        let start = rows.len().div_ceil(size) * size;
        pad(&mut rows, start);
        let mut at = first;
        let per_port = (pa..pb).map(|p| { let n = (at..at + ib[p]).filter(|w| !export_col.contains_key(w)).count(); at += ib[p]; n }).collect();
        in_groups.push((start / WORD, size / WORD, per_port));
        for i in 0..size {
            let c = start + i;
            rows.push(if i < kept.len() { (vec![c as i64], vec![c as i64]) } else { (Vec::new(), Vec::new()) });
        }
        for (i, &w) in kept.iter().enumerate() {
            in_cols[w] = start + i;
            wire[w] = Form { wires: vec![start + i], c: false };
        }
    }
    for (&w, &col) in &export_col {
        in_cols[w] = col;
        wire[w] = Form { wires: vec![col], c: false };
    }
    let (mut steps, mut call_forms) = (Vec::new(), Vec::new());
    for it in &t.items {
        match it {
            Item::And(a, b) => {
                let row = (entries(&xor(&wire, a)), entries(&xor(&wire, b)));
                steps.push(Step::Row(rows.len()));
                wire.push(Form { wires: vec![rows.len()], c: false });
                rows.push(row);
            }
            Item::Call(_, args) | Item::Read(_, args) => {
                let p = call_forms.len();
                let (base, sub) = subs[p];
                call_forms.push(args.iter().map(|f| xor(&wire, f)).collect::<Vec<_>>());
                wire.extend(sub.out_cols.iter().map(|&c| Form { wires: vec![base + c], c: false }));
                steps.push(Step::Part(p));
            }
        }
    }
    let outs: Vec<Form> = t.outputs.iter().map(|f| xor(&wire, f)).collect();
    let ob = bits(&t.outs);
    let (mut out_groups, mut out_cols) = (Vec::new(), Vec::new());
    for (first, count, pa, pb) in groups(&ob, &lay.out_split)? {
        let size = size_of(count);
        let start = rows.len().div_ceil(size) * size;
        pad(&mut rows, start);
        out_groups.push((start / WORD, size / WORD, ob[pa..pb].to_vec()));
        for o in &outs[first..first + count] {
            out_cols.push(rows.len());
            rows.push((entries(o), vec![KONST]));
        }
        pad(&mut rows, start + size);
    }
    let konst = rows.len();
    rows.push((vec![KONST], vec![KONST]));
    let resolve = |m: Vec<i64>| {
        let mut v: Vec<usize> = m.into_iter().map(|x| if x == KONST { konst } else { x as usize }).collect();
        v.sort_unstable();
        v
    };
    let binds = call_forms.into_iter().zip(subs).zip(&exported).map(|((forms, &(base, sub)), ex)| {
        forms.into_iter().enumerate().filter(|(bit, _)| !ex.contains(bit)).map(|(bit, f)| (base + sub.in_cols[bit], resolve(entries(&f)))).collect()
    }).collect();
    let rows = rows.into_iter().map(|(a, b)| (resolve(a), resolve(b))).collect();
    Ok((Rows { rows, konst, in_groups, out_groups, in_cols, out_cols, read: None }, steps, binds))
}

/// The rows as `verity_flock.ir_lower.netlist` writes a unit (`flock-ir-unit/v2`, no `LEAVES` or `CUT` line), then a
/// generated read's `READ` line (`verity_flock.derive.netlist`); entries at or past `cols` (another slot's columns) left out.
pub fn netlist(d: &Rows, name: &str, cols: usize) -> String {
    let mut head = vec!["flock-ir-unit/v2".to_string(), name.to_string(), d.rows.len().to_string(), d.konst.to_string()];
    for groups in [&d.in_groups, &d.out_groups] {
        head.push(groups.len().to_string());
        for (word, words, bits) in groups {
            head.extend([*word, *words, bits.len()].iter().chain(bits).map(usize::to_string));
        }
    }
    let mut out = head.join(" ") + "\n";
    for (a, b) in &d.rows {
        let (a, b): (Vec<usize>, Vec<usize>) = (a.iter().copied().filter(|&x| x < cols).collect(), b.iter().copied().filter(|&x| x < cols).collect());
        let line: Vec<String> = std::iter::once(a.len()).chain(a).chain(std::iter::once(b.len())).chain(b).map(|x| x.to_string()).collect();
        out += &line.join(" ");
        out.push('\n');
    }
    if let Some((sha, n, lo, k, prod0, lo_top)) = &d.read {
        out += &format!("READ {sha} {n} {lo} {k} {prod0} {lo_top}\n");
    }
    out
}

fn head_checks(meta: &Value) -> Result<(), String> {
    if meta["policy"] != POLICY {
        return Err(format!("META: the layout policy is not {POLICY}"));
    }
    if meta["unit_in"] != json!([]) {
        return Err("META: unit_in binds a unit input to a constant: not read yet".into());
    }
    Ok(())
}

/// Whether a typed statement's unit is a flat class's: one type, under the one layout.
pub fn is_flat(meta: &Value) -> bool {
    meta["layouts"].as_array().is_some_and(|l| l.len() == 1) && meta["types"].as_object().is_some_and(|t| t.len() == 1)
}

/// The unit's rows of a flat class's typed statement `META`, as the `flock-ir-unit/v2` text of a unit named `name`. Its
/// type and layout are held to their digests.
pub fn unit_rows(meta: &Value, name: &str) -> Result<String, String> {
    head_checks(meta)?;
    let types = types(meta)?;
    let layouts = meta["layouts"].as_array().ok_or("META: layouts")?;
    if !is_flat(meta) || meta["unit_layout"] != 0 {
        return Err("META: more than the unit's type and layout (a template with a tail)".into());
    }
    let lay = Layout::parse(&layouts[0])?;
    if meta["unit_type"].as_str() != lay.ty.as_deref() || lay.range_log.is_some() || !lay.calls.is_empty() || !lay.reads.is_empty() {
        return Err("META: the unit's layout is not of unit_type, or it has a range or placed calls or reads".into());
    }
    let t = types.get(lay.ty.as_deref().unwrap_or("")).ok_or("META: the unit's type is not among its types")?;
    if !t.calls.is_empty() || !t.tables.is_empty() {
        return Err("META: a flat unit's type makes calls or reads tables".into());
    }
    Ok(netlist(&own_region(t, &lay, &[])?.0, name, usize::MAX))
}

/// A template's instance, typed: its unit's own region and every placed part, in unit coordinates, with the block's Δ
/// between them.
#[derive(Clone)]
pub struct Instance {
    /// the unit type's ports
    pub ins: Vec<(usize, usize)>,
    pub outs: Vec<(usize, usize)>,
    /// the unit's own region, `[0, own)`; a row may read a part's columns
    pub root: Rows,
    /// `(layout index, base)` per placed call and read, in item order
    pub parts: Vec<(usize, usize)>,
    /// each placed layout: its digest, its range's log size, its rows
    pub layouts: BTreeMap<usize, (String, usize, Rows)>,
    /// per part, its bound input rows and their sources (every other input row is exported: an input of the unit)
    pub bind: Vec<Vec<(usize, Vec<usize>)>>,
    /// each generated read layout's slot, whose fast witness fills its rows
    pub reads: BTreeMap<usize, LookupNet>,
    steps: Vec<Step>,
    /// the unit coordinates' end
    pub extent: usize,
    /// the AND items of the unit's type and of every part's (a read's: its rows with an entry outside its words)
    pub ands: usize,
    inputs: BTreeMap<usize, Vec<bool>>,
}

/// The table named by `sha`, of `2^n` words, from `dir`: the file `<sha>` (its raw u32 LE words, an archive's `sha512/`), or
/// for a library table (`lookup::LIBRARY`) its `<name>.u32` (held to its SHA-256 pin); either held to that SHA-512.
fn held(dir: Option<&str>, sha: &str, n: usize) -> Result<&'static [u32], String> {
    // once per process: every parse of the statement reads the same words
    static HELD: std::sync::OnceLock<std::sync::Mutex<BTreeMap<String, &'static [u32]>>> = std::sync::OnceLock::new();
    let cache = HELD.get_or_init(Default::default);
    if let Some(&w) = cache.lock().unwrap().get(sha) {
        return Ok(w);
    }
    let dir = dir.ok_or("the statement reads tables: pass --tables DIR")?;
    let words: Vec<u32> = match std::fs::read(format!("{dir}/{sha}")) {
        Ok(b) => b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
        Err(_) => {
            let &(name, ..) = crate::lookup::LIBRARY.iter().find(|t| t.1 == sha).ok_or(format!("table {}.. is not held under {dir}", &sha[..16]))?;
            let t = crate::ir_tail::Tables::load(dir, &BTreeSet::from([name]))?;
            match name { "rsq" => t.rsq, "sqrt" => t.sqrt, "rcp" => t.rcp, "ex2" => t.ex2, _ => Vec::new() }
        }
    };
    if words.len() != 1 << n || crate::lookup::table_sha512(&words) != sha {
        return Err(format!("the held table {}.. is not the one the statement names", &sha[..16]));
    }
    let w: &'static [u32] = Box::leak(words.into_boxed_slice());
    cache.lock().unwrap().insert(sha.to_string(), w);
    Ok(w)
}

/// A generated read's slot, `LookupNet::build_v2` over its held table: its rows with the product rows' B side left to the
/// table (as `verity_flock.derive.gen_read` lists them, with the `READ` record), the slot's net for the block (the same
/// rows), its fast witness, and its AND rows (every row with an entry, outside its words and constant).
fn gen_rows(name: &str, sha: &str, words: &'static [u32], n: usize, lo: usize, bits: usize) -> Result<(Rows, IrUnitNet, LookupNet, usize), String> {
    // once per process and shape: `build_v2` holds the table's side as explicit rows until they're cleared here
    type Slot = (Rows, IrUnitNet, LookupNet, usize);
    static BUILT: std::sync::OnceLock<std::sync::Mutex<BTreeMap<(String, usize, usize, usize), Slot>>> = std::sync::OnceLock::new();
    let key = (sha.to_string(), n, lo, bits);
    if let Some(x) = BUILT.get_or_init(Default::default).lock().unwrap().get(&key) {
        return Ok(x.clone());
    }
    let (mut net, ln) = LookupNet::build_v2(name, sha, words, n, lo, bits)?;
    for r in ln.prod0()..ln.prod0() + (ln.k << (n - lo)) {
        net.b[r] = Vec::new();
    }
    let group = |g: &crate::ir_block::Group| (g.col, g.words, g.bits.clone());
    let out0 = net.out_groups[0].col * WORD;
    let rows = Rows {
        rows: net.a.iter().cloned().zip(net.b.iter().cloned()).collect(), konst: net.const_pos,
        in_groups: net.in_groups.iter().map(group).collect(), out_groups: net.out_groups.iter().map(group).collect(),
        in_cols: (0..n).collect(), out_cols: (out0..out0 + bits).collect(), read: Some((sha.to_string(), n, lo, ln.k, ln.prod0(), ln.lo_top())),
    };
    let word = |r: usize| r < WORD || (out0..out0 + WORD).contains(&r);
    let ands = (0..net.useful).filter(|&r| r != net.const_pos && !word(r) && !(net.a[r].is_empty() && net.b[r].is_empty())).count();
    let slot = (rows, net, ln, ands);
    BUILT.get_or_init(Default::default).lock().unwrap().insert(key, slot.clone());
    Ok(slot)
}

/// The typed instance of a template statement's `META`, and each generated read layout's slot net: its types and layouts
/// held to their digests, the unit's layout placing every call and read of its type, a call in a range of its callee's
/// layout (which places nothing), a read in a range of its table's generated layout. `tables` holds the tables the reads
/// name.
pub fn instance(meta: &Value, tables: Option<&str>) -> Result<(Instance, BTreeMap<usize, IrUnitNet>), String> {
    head_checks(meta)?;
    let types = types(meta)?;
    let layouts = meta["layouts"].as_array().ok_or("META: layouts")?.iter().map(Layout::parse).collect::<Result<Vec<_>, _>>()?;
    let u = uint(&meta["unit_layout"], "META: unit_layout")?;
    let lay = layouts.get(u).ok_or("META: unit_layout is not a layout")?;
    if meta["unit_type"].as_str() != lay.ty.as_deref() || lay.range_log.is_some() || lay.own != Own::Derive {
        return Err("META: the unit's layout is not a derived region of unit_type, placed by the block".into());
    }
    let t = types.get(lay.ty.as_deref().unwrap_or("")).ok_or("META: the unit's type is not among its types")?;
    let items = |call: bool| -> Vec<(usize, usize)> {
        t.items.iter().enumerate().filter_map(|(i, it)| match it {
            Item::Call(k, _) if call => Some((i, *k)),
            Item::Read(k, _) if !call => Some((i, *k)),
            _ => None,
        }).collect()
    };
    let (call_items, read_items) = (items(true), items(false));
    if lay.calls.len() != call_items.len() || lay.reads.len() != read_items.len() {
        return Err("META: the unit's layout does not place every call and read of its type".into());
    }
    let (mut placed, mut layout_ands, mut slot, mut reads, mut nets) = (BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    for (call, entries, its) in [(true, &lay.calls, &call_items), (false, &lay.reads, &read_items)] {
        for (&(item, j, q), &(i, k)) in entries.iter().zip(its) {
            let sub = layouts.get(j).filter(|_| j != u).ok_or("META: an entry placed by no layout (or by the unit's)")?;
            let s = slot.entry(j).or_insert(0usize);
            let fits = if call {
                sub.own == Own::Derive && sub.ty.as_deref() == Some(t.calls[k].as_str()) && sub.calls.is_empty() && sub.reads.is_empty()
            } else {
                let (sha, ib, vb) = &t.tables[k];
                matches!(&sub.own, Own::Gen { table, in_bits, value_bits, .. } if table == sha && in_bits == ib && value_bits == vb)
            };
            if item != i || q != *s || !fits {
                return Err("META: an entry is not its item's, at its slot, in its callee's layout or its table's generated one".into());
            }
            *s += 1;
            if let std::collections::btree_map::Entry::Vacant(e) = placed.entry(j) {
                let range_log = sub.range_log.ok_or("META: a placed layout has no range")?;
                let (rows, ands) = match &sub.own {
                    Own::Derive => {
                        let st = types.get(sub.ty.as_deref().unwrap_or("")).ok_or("META: a callee's type is not among its types")?;
                        let (rows, steps, _) = own_region(st, sub, &[])?;
                        (rows, steps.len())
                    }
                    Own::Gen { table, in_bits, value_bits, lo_bits } => {
                        let (rows, net, ln, ands) = gen_rows(&sub.sha512[..16], table, held(tables, table, *in_bits)?, *in_bits, *lo_bits, *value_bits)?;
                        reads.insert(j, ln);
                        nets.insert(j, net);
                        (rows, ands)
                    }
                };
                if rows.rows.len() > 1 << range_log {
                    return Err("META: a placed layout's rows overrun its range".into());
                }
                layout_ands.insert(j, ands);
                e.insert((sub.sha512.clone(), range_log, rows));
            }
        }
    }
    if layouts.len() != placed.len() + 1 {
        return Err("META: a layout the unit places nowhere".into());
    }
    // every placed entry, in item order: the parts
    let mut entries: Vec<(usize, usize, usize)> = lay.calls.iter().chain(&lay.reads).copied().collect();
    entries.sort_unstable();
    let at = |bases: &[usize]| -> Vec<(usize, &Rows)> { entries.iter().zip(bases).map(|(&(_, j, _), &b)| (b, &placed[&j].2)).collect() };
    let own = own_region(t, lay, &at(&vec![0; entries.len()]))?.0.rows.len();
    let (mut bases, mut cursor) = (Vec::new(), own);
    for &(_, j, _) in &entries {
        let span = 1usize << placed[&j].1;
        let base = cursor.div_ceil(span) * span;
        bases.push(base);
        cursor = base + span;
    }
    let (root, steps, bind) = own_region(t, lay, &at(&bases))?;
    let parts: Vec<(usize, usize)> = entries.iter().zip(&bases).map(|(&(_, j, _), &b)| (j, b)).collect();
    let inputs = placed.iter().map(|(&j, (_, _, r))| {
        let mut v = vec![false; r.rows.len()];
        r.in_cols.iter().for_each(|&c| v[c] = true);
        (j, v)
    }).collect();
    let ands = steps.iter().filter(|s| matches!(s, Step::Row(_))).count() + parts.iter().map(|(j, _)| layout_ands[j]).sum::<usize>();
    let inst = Instance { ins: t.ins.clone(), outs: t.outs.clone(), root, parts, layouts: placed, bind, reads, steps, extent: cursor.max(own), ands, inputs };
    inst.check_columns()?;
    Ok((inst, nets))
}

/// The rows the prover derives, as `verity_flock.typed_statement.stage` writes them beside the statement (`rows/`):
/// `unit.txt` for a flat class; for a template, `root.txt` (in unit coordinates), `<layout digest>.txt` per placed layout
/// and `delta.txt` (`verity_flock.derive.delta_text`).
pub fn staged_rows(meta: &Value, name: &str, tables: Option<&str>) -> Result<Vec<(String, String)>, String> {
    if is_flat(meta) {
        return Ok(vec![("unit.txt".into(), unit_rows(meta, name)?)]);
    }
    let (inst, _) = instance(meta, tables)?;
    let mut out = vec![("root.txt".to_string(), netlist(&inst.root, name, usize::MAX))];
    out.extend(inst.layouts.iter().map(|(&j, (d, _, _))| (format!("{d}.txt"), inst.layout_text(j))));
    let mut delta: Vec<String> = inst.parts.iter().map(|&(j, base)| format!("PART {} {base}", inst.layouts[&j].0)).collect();
    delta.push(format!("DELTA {} 1 one", inst.root.konst));
    for (&(j, base), binds) in inst.parts.iter().zip(&inst.bind) {
        delta.push(format!("DELTA {} 1 one", base + inst.layouts[&j].2.konst));
        delta.extend(binds.iter().map(|(r, src)| format!("DELTA {r} {} {}", src.len(), src.iter().map(usize::to_string).collect::<Vec<_>>().join(" ")).trim_end().to_string()));
    }
    out.push(("delta.txt".into(), delta.join("\n") + "\n"));
    Ok(out)
}

/// `meta` with its unit type's first two outputs swapped, the type re-keyed and its layout re-digested: a relabelled
/// statement over the same rows.
pub fn relabel(meta: &mut Value) -> Result<(), String> {
    let ut = meta["unit_type"].as_str().ok_or("META: unit_type")?.to_string();
    let mut content = meta["types"][&ut].clone();
    content["outputs"].as_array_mut().filter(|o| o.len() >= 2).ok_or("the unit type has fewer than two outputs")?.swap(0, 1);
    let d = type_digest(&content);
    let types = meta["types"].as_object_mut().ok_or("META: types")?;
    types.remove(&ut);
    types.insert(d.clone(), content);
    meta["unit_type"] = json!(d);
    let u = uint(&meta["unit_layout"], "META: unit_layout")?;
    let lay = &mut meta["layouts"][u];
    lay["type"] = json!(d);
    lay["sha512"] = json!(layout_digest(lay));
    Ok(())
}

impl Instance {
    /// Whether unit coordinate `u` lies in the own region or in a part's rows (not its range's padding, not past the last).
    pub fn inside(&self, u: usize) -> bool {
        u < self.root.rows.len() || self.parts.iter().any(|&(j, base)| (base..base + self.layouts[&j].2.rows.len()).contains(&u))
    }

    /// Every column the block reads through the instance lies inside it: the own rows' entries (the cross entries
    /// among them), the inputs' columns, and each binding's row (a part's input row) and sources.
    fn check_columns(&self) -> Result<(), String> {
        let bad = |what: &str| Err(format!("META: {what} outside the unit's own region and its parts' rows"));
        if self.root.rows.iter().any(|(a, b)| a.iter().chain(b).any(|&x| !self.inside(x))) {
            return bad("an own row's entry");
        }
        if self.root.in_cols.iter().any(|&x| !self.inside(x)) {
            return bad("an input's column");
        }
        for (&(j, base), binds) in self.parts.iter().zip(&self.bind) {
            for (r, src) in binds {
                if !(base..base + self.inputs[&j].len()).contains(r) || !self.inputs[&j][r - base] {
                    return Err("META: a binding's row is not an input row of its part".into());
                }
                if src.iter().any(|&x| !self.inside(x)) {
                    return bad("a binding's source");
                }
            }
        }
        Ok(())
    }

    /// The first AND row (unit coordinates): the own region's, else a part's.
    pub fn first_and(&self) -> Option<usize> {
        self.steps.iter().find_map(|s| match s { Step::Row(c) => Some(*c), _ => None }).or_else(|| {
            self.parts.iter().find_map(|&(j, base)| {
                let (rows, is_in) = (&self.layouts[&j].2, &self.inputs[&j]);
                (0..rows.rows.len()).find(|&c| c != rows.konst && !is_in[c] && !rows.rows[c].0.is_empty()).map(|c| base + c)
            })
        })
    }

    /// The own region's rows (`[0, own)`), every entry at a part's column left out.
    pub fn root_text(&self, name: &str) -> String {
        netlist(&self.root, name, self.root.rows.len())
    }

    /// Placed layout `j`'s rows.
    pub fn layout_text(&self, j: usize) -> String {
        let (d, _, rows) = &self.layouts[&j];
        netlist(rows, &d[..16], usize::MAX)
    }

    /// The own rows' entries at parts' columns (unit coordinates): `(row, A's, B's)`.
    pub fn cross(&self) -> Vec<(usize, Vec<usize>, Vec<usize>)> {
        let own = self.root.rows.len();
        self.root.rows.iter().enumerate().filter_map(|(r, (a, b))| {
            let (a, b): (Vec<usize>, Vec<usize>) = (a.iter().copied().filter(|&x| x >= own).collect(), b.iter().copied().filter(|&x| x >= own).collect());
            (!a.is_empty() || !b.is_empty()).then_some((r, a, b))
        }).collect()
    }

    /// `(z, A z, B z)` over the unit coordinates for 64 instances at once (lane `l` of each word instance `l`), from each
    /// instance's input bits (`input(w)`: bit `w` of every lane): the constants 1, the inputs at their columns, then the
    /// items in order (a part: its bound inputs from their sources, then its rows), then the output copies. `flip`
    /// `(row, lanes)` flips those lanes of a bound input row once it's bound (a witness that breaks only that binding).
    pub fn eval64(&self, input: &dyn Fn(usize) -> u64, flip: Option<(usize, u64)>) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let n = self.extent;
        let (mut z, mut a, mut b) = (vec![0u64; n], vec![0u64; n], vec![0u64; n]);
        let set = |z: &mut [u64], a: &mut [u64], b: &mut [u64], c: usize, v: u64| { z[c] = v; a[c] = v; b[c] = v; };
        set(&mut z, &mut a, &mut b, self.root.konst, !0);
        for &(j, base) in &self.parts {
            set(&mut z, &mut a, &mut b, base + self.layouts[&j].2.konst, !0);
        }
        for (w, &c) in self.root.in_cols.iter().enumerate() {
            set(&mut z, &mut a, &mut b, c, input(w));
        }
        let row = |z: &mut [u64], a: &mut [u64], b: &mut [u64], at: usize, (ra, rb): &(Vec<usize>, Vec<usize>), off: usize| {
            let (va, vb) = (ra.iter().fold(0, |x, &e| x ^ z[off + e]), rb.iter().fold(0, |x, &e| x ^ z[off + e]));
            a[at] = va;
            b[at] = vb;
            z[at] = va & vb;
        };
        for &st in &self.steps {
            match st {
                Step::Row(c) => row(&mut z, &mut a, &mut b, c, &self.root.rows[c], 0),
                Step::Part(p) => {
                    for (r, src) in &self.bind[p] {
                        let v = src.iter().fold(0, |x, &e| x ^ z[e]) ^ flip.filter(|f| f.0 == *r).map_or(0, |f| f.1);
                        set(&mut z, &mut a, &mut b, *r, v);
                    }
                    let (j, base) = self.parts[p];
                    if let Some(ln) = self.reads.get(&j) {
                        // a read: each lane's rows are its fast witness at the index its bound rows hold
                        for l in 0..64 {
                            let idx = (0..ln.n).fold(0usize, |x, bit| x | ((((z[base + bit] >> l) & 1) as usize) << bit));
                            let (wz, wa, wb) = ln.witness(idx);
                            for (dst, src) in [(&mut z, &wz), (&mut a, &wa), (&mut b, &wb)] {
                                for (w, &word) in src.iter().enumerate() {
                                    let mut x = word;
                                    while x != 0 {
                                        dst[base + 64 * w + x.trailing_zeros() as usize] |= 1 << l;
                                        x &= x - 1;
                                    }
                                }
                            }
                        }
                        continue;
                    }
                    let (rows, is_in) = (&self.layouts[&j].2, &self.inputs[&j]);
                    for (c, rw) in rows.rows.iter().enumerate() {
                        if c != rows.konst && !is_in[c] {
                            row(&mut z, &mut a, &mut b, base + c, rw, base);
                        }
                    }
                }
            }
        }
        for &c in &self.root.out_cols {
            row(&mut z, &mut a, &mut b, c, &self.root.rows[c], 0);
        }
        (z, a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Sha256;

    /// A flat type of two input groups, an AND with a constant form, an AND over an XOR and two output ports in one group,
    /// with `verity_flock.circuit_types`' digest, `verity_flock.layouts`' record and the SHA-256 of `verity_flock.derive`'s
    /// netlist of it (unit name `t`).
    const CONTENT: &str = r#"{"calls":[],"format":"verity/circuit-type/v1","in":[[1,2],[1,1]],"items":[[0,[[0],0],[[1],1]],[0,[[2,3],0],[[0],0]]],"label":"program","out":[[1,1],[1,1]],"outputs":[[[3],0],[[0,2,4],1]],"tables":[]}"#;
    const TYPE: &str = "3ca26baa5024af3ea96a08e63ea90814ab025eb6b59291b3d7fe17548e496aa243742fba9aba08fc2bcd35388bdf5d89823648c69d9378b81af1986924c53dd7";
    const LAYOUT: &str = r#"{"calls":[],"in_split":[1],"out_split":[],"own":{"derive":"v2"},"reads":[],"sha512":"fa46a775a07166964967f473598baf43517481d05100c0d259bfc489dc14abc1abedba107b1827ff69d0dd7921ddc23477d3cd0b0074ca79dd5c4b06539b8d7e","type":"3ca26baa5024af3ea96a08e63ea90814ab025eb6b59291b3d7fe17548e496aa243742fba9aba08fc2bcd35388bdf5d89823648c69d9378b81af1986924c53dd7"}"#;
    const ROWS_SHA256: &str = "adf55dcfcef23d628f866b8017b2c1ae81233ef691ce9c9135936955434c5eb6";

    fn meta() -> Value {
        let (content, layout): (Value, Value) = (serde_json::from_str(CONTENT).unwrap(), serde_json::from_str(LAYOUT).unwrap());
        json!({"policy": POLICY, "unit_in": [], "types": {TYPE: content}, "layouts": [layout], "unit_layout": 0, "unit_type": TYPE})
    }

    #[test]
    fn derives_the_python_rows() {
        let text = unit_rows(&meta(), "t").unwrap();
        assert_eq!(crate::hex(&Sha256::digest(text.as_bytes())), ROWS_SHA256);
        assert!(text.starts_with("flock-ir-unit/v2 t 513 512 2 0 1 1 2 1 1 1 1 1 3 1 2 1 1\n"));
    }

    /// A unit of three input bits calling a flat type once, placed at 512: its arguments an input an output also reads
    /// (bound, not exported), an AND and a constant form. With the SHA-256s of `verity_flock.derive`'s root rows, part
    /// rows and Δ (`delta_text`), and `circuit_types.evaluate`'s outputs on all eight inputs (bit-sliced).
    const TAIL: &str = r#"{"layouts":[{"calls":[],"in_split":[],"out_split":[],"own":{"derive":"v2"},"range_log":9,"reads":[],"sha512":"b8197d84909856109ee92dbe7160616e300e46d75a4087ab8837992ffbe6e0001ded51e55ce4091f8b22b4b339a31298ee067054c3ab15a56baf146df60aebfb","type":"724c5adfcf6cd5dec85feb6f3831024a24b5009f6d227596f576e7a804e874bd8762b442b0ff81786343dd4173be160985d9211803c9542b59913bbda3ece894"},{"calls":[{"item":1,"layout":0,"offset":0}],"in_split":[],"out_split":[],"own":{"derive":"v2"},"reads":[],"sha512":"d3762b6fb5d12b01fb4ffcdb39dbd37de5718596bfa052be5d46b72999eaffacce349ba41cfca1251d17c5821025f7d6be02879a4457fda153ae9b1310753297","type":"90cbb358bf76988fe7a7991e6ebbf1c24c2f0a8684b8acffa3a6887ea874ec9e0b7dfd9e9bf2ab58493e60cc4ecfc8eded425314611e4f315f47c81f5357a9b8"}],"policy":"calls/v1","types":{"724c5adfcf6cd5dec85feb6f3831024a24b5009f6d227596f576e7a804e874bd8762b442b0ff81786343dd4173be160985d9211803c9542b59913bbda3ece894":{"calls":[],"format":"verity/circuit-type/v1","in":[[1,2],[1,1]],"items":[[0,[[0],0],[[1],0]],[0,[[3],0],[[2],1]]],"label":"program","out":[[1,1]],"outputs":[[[4],0]],"tables":[]},"90cbb358bf76988fe7a7991e6ebbf1c24c2f0a8684b8acffa3a6887ea874ec9e0b7dfd9e9bf2ab58493e60cc4ecfc8eded425314611e4f315f47c81f5357a9b8":{"calls":["724c5adfcf6cd5dec85feb6f3831024a24b5009f6d227596f576e7a804e874bd8762b442b0ff81786343dd4173be160985d9211803c9542b59913bbda3ece894"],"format":"verity/circuit-type/v1","in":[[1,3]],"items":[[0,[[0],0],[[1],0]],[3,0,[[[2],0],[[3],0],[[0],1]]],[0,[[4],0],[[1],0]]],"label":"program","out":[[1,2]],"outputs":[[[5],0],[[2,4],0]],"tables":[]}},"unit_in":[],"unit_layout":1,"unit_type":"90cbb358bf76988fe7a7991e6ebbf1c24c2f0a8684b8acffa3a6887ea874ec9e0b7dfd9e9bf2ab58493e60cc4ecfc8eded425314611e4f315f47c81f5357a9b8"}"#;
    const TAIL_SHA256: [&str; 3] = ["fb4dadd1f24e22ed837a6f4d52959bb72ed52fd07e7625a80998b0a42795744f",
                                    "1c8172f7c858550c30c9f8c11ce6ccfa063b3238bef0eb25d006cf974565faab",
                                    "1ad9598dde01e5ebe72c8f7df08dd2c747234130545214ba2fa2db2becf987b7"];

    #[test]
    fn derives_and_evaluates_a_placed_call() {
        let meta: Value = serde_json::from_str(TAIL).unwrap();
        let rows = staged_rows(&meta, "t", None).unwrap();
        assert_eq!(rows.iter().map(|r| r.0.as_str()).collect::<Vec<_>>(),
                   ["root.txt", &format!("{}.txt", meta["layouts"][0]["sha512"].as_str().unwrap()), "delta.txt"]);
        assert_eq!(rows.iter().map(|(_, t)| crate::hex(&Sha256::digest(t.as_bytes()))).collect::<Vec<_>>(), TAIL_SHA256);
        let (inst, _) = instance(&meta, None).unwrap();
        let (z, a, b) = inst.eval64(&|w| (0..8).fold(0, |x, l| x | (((l >> w) & 1) << l)), None);
        assert_eq!(inst.root.out_cols.iter().map(|&c| z[c] & 0xff).collect::<Vec<_>>(), [0x80, 0x70]);
        assert!(z.iter().zip(&a).zip(&b).all(|((z, a), b)| *z == a & b));
        // a bound input flipped in lane 0: every row still holds, the binding alone breaks
        let r = inst.bind[0][0].0;
        let (z2, a2, b2) = inst.eval64(&|w| (0..8).fold(0, |x, l| x | (((l >> w) & 1) << l)), Some((r, 1)));
        assert!(z2.iter().zip(&a2).zip(&b2).all(|((z, a), b)| *z == a & b) && z2[r] == z[r] ^ 1);
    }

    /// A unit of 8 input bits reading a program table of 2^8 words (5 value bits) at them, placed in its generated layout's
    /// range: value bits 0 and 1 ANDed, bits 2 and 4 XORed, on 64 indices; every row holds, and the read's rows are listed
    /// with its product rows' B side empty and its `READ` record.
    #[test]
    fn derives_and_evaluates_a_placed_read() {
        let table: Vec<u32> = (0..256u32).map(|i| (i * 37 + 11) % 32).collect();
        let sha = crate::lookup::table_sha512(&table);
        let dir = std::env::temp_dir().join(format!("flock-typed-read-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(&sha), table.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let root = json!({"format": TYPE_FORMAT, "label": "program", "in": [[1, 8]], "out": [[1, 2]], "calls": [], "tables": [[sha, 8, 5]],
                          "items": [[4, 0, (0..8).map(|w| json!([[w], 0])).collect::<Vec<_>>()], [0, [[8], 0], [[9], 0]]],
                          "outputs": [[[13], 0], [[10, 12], 0]]});
        let r = type_digest(&root);
        let with_digest = |mut l: Value| { l["sha512"] = json!(layout_digest(&l)); l };
        let read_layout = with_digest(json!({"range_log": 10, "in_split": [], "out_split": [], "calls": [], "reads": [],
                                     "own": {"gen": "table/v2", "table": sha, "in_bits": 8, "value_bits": 5, "lo_bits": 4}}));
        let unit = with_digest(json!({"type": r, "in_split": [], "out_split": [], "own": {"derive": "v2"}, "calls": [],
                                      "reads": [{"item": 0, "layout": 0, "offset": 0}]}));
        let meta = json!({"policy": POLICY, "unit_in": [], "types": {r.clone(): root}, "layouts": [read_layout, unit], "unit_layout": 1, "unit_type": r});
        let (inst, nets) = instance(&meta, Some(dir.to_str().unwrap())).unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        let idx: Vec<usize> = (0..64).map(|l| (l * 5 + 3) % 256).collect();
        let (z, a, b) = inst.eval64(&|w| idx.iter().enumerate().fold(0, |x, (l, &i)| x | ((((i >> w) & 1) as u64) << l)), None);
        let want = |f: &dyn Fn(u32) -> u32| idx.iter().enumerate().fold(0u64, |x, (l, &i)| x | ((f(table[i]) & 1) as u64) << l);
        assert_eq!(z[inst.root.out_cols[0]], want(&|v| v & (v >> 1)));
        assert_eq!(z[inst.root.out_cols[1]], want(&|v| (v >> 2) ^ (v >> 4)));
        assert!(z.iter().zip(&a).zip(&b).all(|((z, a), b)| *z == a & b));
        let (_, _, rows) = &inst.layouts[&0];
        let (_, n, lo, k, prod0, lo_top) = rows.read.clone().unwrap();
        assert_eq!((n, lo, k), (8, 4, 5), "a program table keeps every value bit");
        assert!(inst.layout_text(0).ends_with(&format!("READ {sha} 8 4 5 {prod0} {lo_top}\n")));
        assert!((prod0..prod0 + (k << (n - lo))).all(|r| rows.rows[r].1.is_empty() && nets[&0].b[r].is_empty()));
        assert_eq!(inst.bind[0].len(), 8, "a read's index is bound, never exported");
    }

    #[test]
    fn refuses_columns_outside_the_instance() {
        let (inst, _) = instance(&serde_json::from_str(TAIL).unwrap(), None).unwrap();
        inst.check_columns().unwrap();
        let (j, base) = inst.parts[0];
        let padding = base + inst.layouts[&j].2.rows.len();
        let mut bad = inst.clone();
        bad.root.rows[inst.root.out_cols[0]].0.push(padding);
        assert!(bad.check_columns().unwrap_err().contains("an own row's entry"));
        let mut bad = inst.clone();
        bad.bind[0][0].1.push(inst.extent);
        assert!(bad.check_columns().unwrap_err().contains("a binding's source"));
    }

    #[test]
    fn refuses_a_type_or_layout_not_its_digest() {
        let mut m = meta();
        m["types"][TYPE]["items"][0][2][1] = json!(0);
        assert!(unit_rows(&m, "t").unwrap_err().contains("digest is not its content's"));
        let mut m = meta();
        m["layouts"][0]["in_split"] = json!([]);
        assert!(unit_rows(&m, "t").unwrap_err().contains("sha512 is not its record's digest"));
        let mut m = meta();
        m["unit_in"] = json!([[0, "run"]]);
        assert!(unit_rows(&m, "t").is_err());
    }
}
