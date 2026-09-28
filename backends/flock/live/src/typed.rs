//! `verity/flock-circuit/types`: the circuit statement whose unit is circuit types under layouts, never rows
//! (`verity_flock.typed_statement`; Project doc `constant-api-public.md` §2.2). The verifier derives the unit's rows from
//! them (the Lean `Flock.Derive`). This module is the Rust prover's mirror of that derivation, as `verity_flock.derive` is
//! Python's. It writes the rows as the `flock-ir-unit/v2` text the flat statement's unit section holds, so the rest of the
//! statement reads as the flat statement's.
//!
//! It reads a flat class: the unit one type of AND items, under a layout that places nothing. Its rows are the v2 own
//! region: the input port groups (aligned power-of-two runs of 128-bit words), one row per AND item, the output port
//! groups' copy rows, and the constant last.

use serde_json::{Value, json};
use sha2::{Digest, Sha512};

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

/// A flat `verity/circuit-type/v1` type: its ports, AND items and outputs.
#[derive(Debug)]
pub struct Type {
    pub ins: Vec<(usize, usize)>,
    pub outs: Vec<(usize, usize)>,
    ands: Vec<(Form, Form)>,
    outputs: Vec<Form>,
}

/// A `verity/layout/v1` record that places nothing: its type and port groups.
#[derive(Debug)]
pub struct Layout {
    pub ty: String,
    pub in_split: Vec<usize>,
    pub out_split: Vec<usize>,
}

/// The rows of a layout's own region, the constant resolved.
#[derive(Debug)]
pub struct Rows {
    pub rows: Vec<(Vec<usize>, Vec<usize>)>,
    pub konst: usize,
    /// per group: first word, words, each port's bits
    pub in_groups: Vec<(usize, usize, Vec<usize>)>,
    pub out_groups: Vec<(usize, usize, Vec<usize>)>,
}

fn sorted(v: &Value) -> Value {
    match v {
        Value::Object(m) => Value::Object(m.iter().map(|(k, x)| (k.clone(), sorted(x))).collect::<std::collections::BTreeMap<_, _>>().into_iter().collect()),
        Value::Array(a) => Value::Array(a.iter().map(sorted).collect()),
        x => x.clone(),
    }
}

/// `SHA-512(domain || 0 || canon(v))`, canon RFC 8785 JSON: for a statement's values (integers, ASCII strings, arrays,
/// objects), keys sorted and no whitespace.
fn digest(domain: &str, v: &Value) -> String {
    let mut h = Sha512::new();
    h.update(domain.as_bytes());
    h.update([0u8]);
    h.update(serde_json::to_string(&sorted(v)).unwrap().as_bytes());
    crate::hex(&h.finalize())
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
    /// The type `content` encodes, a flat one: no calls, no tables, AND items only.
    pub fn parse(content: &Value) -> Result<Self, String> {
        let keys = ["format", "label", "in", "out", "calls", "tables", "items", "outputs"];
        if content["format"] != TYPE_FORMAT || content.as_object().is_none_or(|m| m.len() != keys.len() || keys.iter().any(|k| !m.contains_key(*k))) {
            return Err(format!("a type is not a {TYPE_FORMAT} content"));
        }
        if !["library", "program"].iter().any(|l| content["label"] == *l) {
            return Err("a type's label is neither library nor program".into());
        }
        if content["calls"] != json!([]) || content["tables"] != json!([]) {
            return Err("a type with calls or table reads (a template with a tail): not read yet".into());
        }
        let (ins, outs) = (ports(&content["in"])?, ports(&content["out"])?);
        let mut n: usize = bits(&ins).iter().sum();
        let mut ands = Vec::new();
        for it in content["items"].as_array().ok_or("a type's items")? {
            match it.as_array().map(|a| a.as_slice()) {
                Some([op, a, b]) if op.as_u64() == Some(0) => ands.push((form(a, n)?, form(b, n)?)),
                _ => return Err("an item that is not an AND [0, A, B]: not read yet".into()),
            }
            n += 1;
        }
        let outputs = content["outputs"].as_array().ok_or("a type's outputs")?.iter().map(|f| form(f, n)).collect::<Result<Vec<_>, _>>()?;
        if outputs.len() != bits(&outs).iter().sum::<usize>() {
            return Err("a type's outputs are not its out ports' bits".into());
        }
        Ok(Type { ins, outs, ands, outputs })
    }
}

impl Layout {
    /// The layout `rec` records, one that places nothing (derived own region, no range, no calls or reads); its
    /// `sha512` must be the record's digest.
    pub fn parse(rec: &Value) -> Result<Self, String> {
        let m = rec.as_object().ok_or("a layout is not an object")?;
        let mut r = m.clone();
        r.remove("sha512");
        if rec["sha512"].as_str() != Some(digest(LAYOUT_FORMAT, &Value::Object(r)).as_str()) {
            return Err("a layout's sha512 is not its record's digest".into());
        }
        let keys = ["sha512", "type", "in_split", "out_split", "own", "calls", "reads"];
        if m.len() != keys.len() || keys.iter().any(|k| !m.contains_key(*k)) || rec["own"] != json!({"derive": "v2"})
            || rec["calls"] != json!([]) || rec["reads"] != json!([])
        {
            return Err("a layout that places a range, calls or reads (a template with a tail): not read yet".into());
        }
        let split = |k: &str| rec[k].as_array().ok_or(format!("a layout's {k}"))?.iter().map(|x| uint(x, k)).collect::<Result<Vec<_>, _>>();
        Ok(Layout { ty: rec["type"].as_str().ok_or("a layout's type")?.to_string(), in_split: split("in_split")?, out_split: split("out_split")? })
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
    let mut cols = std::collections::BTreeSet::new();
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

/// A flat type's own region under `lay` (`verity_flock.derive.Derivation`, the v2 rule): `ir_lower.netlist`'s rows.
pub fn derive(t: &Type, lay: &Layout) -> Result<Rows, String> {
    let mut rows: Vec<(Vec<i64>, Vec<i64>)> = Vec::new();
    let pad = |rows: &mut Vec<(Vec<i64>, Vec<i64>)>, to: usize| rows.resize(to, (Vec::new(), Vec::new()));
    let size_of = |bits: usize| WORD * ceil_pow2(bits.div_ceil(WORD));
    let ib = bits(&t.ins);
    let mut wire: Vec<Form> = vec![Form { wires: Vec::new(), c: false }; ib.iter().sum()];
    let mut in_groups = Vec::new();
    for (first, count, pa, pb) in groups(&ib, &lay.in_split)? {
        let size = size_of(count);
        let start = rows.len().div_ceil(size) * size;
        pad(&mut rows, start);
        in_groups.push((start / WORD, size / WORD, ib[pa..pb].to_vec()));
        for i in 0..size {
            let c = start + i;
            rows.push(if i < count { (vec![c as i64], vec![c as i64]) } else { (Vec::new(), Vec::new()) });
            if i < count {
                wire[first + i] = Form { wires: vec![c], c: false };
            }
        }
    }
    for (a, b) in &t.ands {
        let row = (entries(&xor(&wire, a)), entries(&xor(&wire, b)));
        wire.push(Form { wires: vec![rows.len()], c: false });
        rows.push(row);
    }
    let outs: Vec<Form> = t.outputs.iter().map(|f| xor(&wire, f)).collect();
    let ob = bits(&t.outs);
    let mut out_groups = Vec::new();
    for (first, count, pa, pb) in groups(&ob, &lay.out_split)? {
        let size = size_of(count);
        let start = rows.len().div_ceil(size) * size;
        pad(&mut rows, start);
        out_groups.push((start / WORD, size / WORD, ob[pa..pb].to_vec()));
        for o in &outs[first..first + count] {
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
    Ok(Rows { rows: rows.into_iter().map(|(a, b)| (resolve(a), resolve(b))).collect(), konst, in_groups, out_groups })
}

/// The rows as `verity_flock.ir_lower.netlist` writes a unit (`flock-ir-unit/v2`, no `LEAVES` or `CUT` line).
pub fn netlist(d: &Rows, name: &str) -> String {
    let mut head = vec!["flock-ir-unit/v2".to_string(), name.to_string(), d.rows.len().to_string(), d.konst.to_string()];
    for groups in [&d.in_groups, &d.out_groups] {
        head.push(groups.len().to_string());
        for (word, words, bits) in groups {
            head.extend([*word, *words, bits.len()].iter().chain(bits).map(usize::to_string));
        }
    }
    let mut out = head.join(" ") + "\n";
    for (a, b) in &d.rows {
        let line: Vec<String> = std::iter::once(a.len()).chain(a.iter().copied()).chain(std::iter::once(b.len())).chain(b.iter().copied()).map(|x| x.to_string()).collect();
        out += &line.join(" ");
        out.push('\n');
    }
    out
}

/// The unit's rows of a typed statement's `META`, as the `flock-ir-unit/v2` text of a unit named `name`. Its types and
/// layouts are held to their digests; a flat class's statement has one type and one layout, its unit's.
pub fn unit_rows(meta: &Value, name: &str) -> Result<String, String> {
    if meta["policy"] != POLICY {
        return Err(format!("META: the layout policy is not {POLICY}"));
    }
    if meta["unit_in"] != json!([]) {
        return Err("META: unit_in binds a unit input to a constant: not read yet".into());
    }
    let types = meta["types"].as_object().ok_or("META: types")?;
    for (d, content) in types {
        if digest(TYPE_FORMAT, content) != *d {
            return Err(format!("META: type {}..'s digest is not its content's", d.get(..16).unwrap_or(d)));
        }
    }
    let layouts = meta["layouts"].as_array().ok_or("META: layouts")?;
    if layouts.len() != 1 || meta["unit_layout"] != 0 || types.len() != 1 {
        return Err("META: more than the unit's type and layout (a template with a tail): not read yet".into());
    }
    let lay = Layout::parse(&layouts[0])?;
    if meta["unit_type"].as_str() != Some(lay.ty.as_str()) {
        return Err("META: the unit's layout is not of unit_type".into());
    }
    let t = Type::parse(types.get(&lay.ty).ok_or("META: the unit's type is not among its types")?)?;
    Ok(netlist(&derive(&t, &lay)?, name))
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
