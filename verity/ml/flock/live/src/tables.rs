//! Multi-table statements (private-recursion's spec): several tables in one Flock union commitment, each a pinned template
//! (`A_t`, `B_t`, `C = I`) repeated over its blocks, with private glue between them (`glue`). The union proves every table's
//! relation (one zerocheck and one lincheck over `Σ_t sel_t ⊗ A_t`); the glue sumcheck runs after them and its claim joins the
//! one merged opening.
//!
//! A table's local index is its in-block bit `c` (`0..k`) and its block `j` (bits `k..k+ν`). The union lays slot `t` out
//! batch-major, so local bit `ℓ` is address bit `ℓ` (`ℓ < 7`, inside a word), `ℓ + ν` (the rest of `c`) or `7 + ℓ − k` (the block);
//! the address bits above the slot are its prefix.

use flock_core::{
    challenger::Challenger,
    lincheck::{LincheckCircuit, SparseMatrixCircuit, pack_z_lincheck},
    proof::{LinkClaim, R1csProofMergedLigerito},
    r1cs::SparseBinaryMatrix,
    verifier::{FlockVerifyError, verify_ligerito_union_glued},
};
use flock_field::F128;
use flock_prover::{
    pcs::{Commitment, PcsParams},
    prover::{UnionSlotProverInput, prove_fast_ligerito_union_glued},
    schedule::{Registry, TableType},
    union::UnionInstance,
};

use crate::glue::{self, GlueProof, Map, Region, Relation};

/// A table's template: `2^k_log` rows, rows `[0, n_in)` its inputs (glue destinations), `const_pos` the constant one, and every
/// other row below `useful` the AND of its `a` and `b` rows' XORs.
#[derive(Clone, Debug)]
pub struct Template {
    pub k_log: usize,
    pub n_in: usize,
    pub const_pos: usize,
    pub useful: usize,
    pub a: Vec<Vec<usize>>,
    pub b: Vec<Vec<usize>>,
}

impl Template {
    /// A `flock-ir-unit/v2` netlist as a template: its input words' rows are the inputs.
    pub fn from_net(net: &crate::ir_block::IrUnitNet) -> Self {
        Template { k_log: net.unit_log, n_in: net.in_words * 128, const_pos: net.const_pos, useful: net.useful, a: net.a.clone(), b: net.b.clone() }
    }

    /// `c = a · b` in GF(2^128) mod `x^128 + x^7 + x^2 + x + 1`, schoolbook: inputs `a` (rows 0..127) and `b` (128..255), the
    /// constant at 256, every product `a_i b_j` (16,384 ANDs), then `c` (128 rows from 16,896) as the reduced XOR of its products.
    pub fn gf128_mul() -> Self {
        let (prod0, out0) = (512, 512 + 128 * 128);
        let k = 1 << 15;
        let (mut a, mut b) = (vec![Vec::new(); k], vec![Vec::new(); k]);
        let mut terms = vec![Vec::new(); 128];
        for i in 0..128 {
            for j in 0..128 {
                let r = prod0 + 128 * i + j;
                (a[r], b[r]) = (vec![i], vec![128 + j]);
                for d in gf128_reduce(i + j) {
                    terms[d].push(r);
                }
            }
        }
        for (d, t) in terms.into_iter().enumerate() {
            (a[out0 + d], b[out0 + d]) = (t, vec![256]);
        }
        Template { k_log: 15, n_in: 256, const_pos: 256, useful: out0 + 128, a, b }
    }

    /// `(z, A z, B z)` of one block from its input bits.
    pub fn eval(&self, inputs: &[bool]) -> (Vec<bool>, Vec<bool>, Vec<bool>) {
        let k = 1 << self.k_log;
        let (mut z, mut av, mut bv) = (vec![false; k], vec![false; k], vec![false; k]);
        for (i, &x) in inputs.iter().enumerate().take(self.n_in) {
            (z[i], av[i], bv[i]) = (x, x, x);
        }
        let c = self.const_pos;
        (z[c], av[c], bv[c]) = (true, true, true);
        for i in self.n_in..self.useful {
            if i == c {
                continue;
            }
            let x = self.a[i].iter().fold(false, |s, &j| s ^ z[j]);
            let y = self.b[i].iter().fold(false, |s, &j| s ^ z[j]);
            (av[i], bv[i], z[i]) = (x, y, x & y);
        }
        (z, av, bv)
    }

    fn matrix(&self, rows: &[Vec<usize>]) -> SparseBinaryMatrix {
        let k = 1 << self.k_log;
        let mut r: Vec<Vec<usize>> = (0..k).map(|i| if i < self.n_in || i == self.const_pos { vec![i] } else { Vec::new() }).collect();
        for i in self.n_in..self.useful {
            if i != self.const_pos {
                r[i] = rows[i].clone();
            }
        }
        SparseBinaryMatrix { num_rows: k, num_cols: k, rows: r.into() }
    }

    fn table_type(&self) -> TableType {
        let k = 1 << self.k_log;
        TableType {
            k_log: self.k_log,
            useful_bits: self.useful,
            a_0: self.matrix(&self.a),
            b_0: self.matrix(&self.b),
            c_0: SparseBinaryMatrix { num_rows: k, num_cols: k, rows: (0..k).map(|i| vec![i]).collect::<Vec<_>>().into() },
            const_pin: Some(self.const_pos),
            class: Default::default(),
            io_schema: Vec::new(),
        }
    }
}

/// The coefficients `x^d` reduces to mod `x^128 + x^7 + x^2 + x + 1` (each once: XOR parity).
fn gf128_reduce(d: usize) -> Vec<usize> {
    let mut v = vec![false; 256];
    v[d] = true;
    for e in (128..256).rev() {
        if v[e] {
            v[e] = false;
            for t in [7, 2, 1, 0] {
                v[e - 128 + t] ^= true;
            }
        }
    }
    (0..128).filter(|&i| v[i]).collect()
}

/// A sub-cube of a table's local index: `fixed` bits and ordered `free` bits over `0..k_log + b` (its blocks' `2^b`); the local
/// block bits past `b` (the union's capacity `ν`) are zero.
#[derive(Clone, Debug)]
pub struct TableRegion {
    pub table: usize,
    pub fixed: Vec<(usize, bool)>,
    pub free: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Glue {
    pub dst: TableRegion,
    pub src: TableRegion,
    pub map: Map,
}

/// The statement: tables (sorted by `k_log` descending: the union's slot order), each with its block count, and the glue.
pub struct Tables {
    pub templates: Vec<Template>,
    pub counts: Vec<usize>,
    pub glue: Vec<Glue>,
    pub registry: Registry,
    pub nu: usize,
    circuits: Vec<SparseMatrixCircuit<'static>>,
    rels: Vec<Relation>,
    /// seconds in the glue sumcheck: the last proof's (prover), the last verification's (verifier)
    pub glue_s: std::cell::Cell<(f64, f64)>,
}

fn leak<T>(x: T) -> &'static T {
    Box::leak(Box::new(x))
}

pub const FORMAT: &str = "flock-tables";
pub const STATEMENT: &str = "verity/flock-tables";

/// A `flock-tables` file: `flock-tables <name>`, a `META` JSON line, then `CIRCUIT <name> <lines>` sections (`flock-ir-unit/v2`
/// expanded circuits, as in a `flock-circuit` file). META:
///
/// ~~~text
/// {"format": "flock-tables", "statement": "verity/flock-tables",
///  "tables": [{"name": str, "circuit": CIRCUIT name, "blocks": n}, ...],          // the union's slot order: k_log descending
///  "glue": [{"dst": REGION, "src": REGION, "map": "id" | {"perm": [...]} | {"broadcast": [...]}}, ...]}
/// REGION = {"table": name, "fixed": {"<bit>": 0 | 1, ...}, "free": [bit, ...]}   // table-local bits: in-block 0..k, block k..k+b
/// ~~~
///
/// Returns the statement and the file's SHA-512 (its pin).
pub fn parse(text: &str) -> Result<(Tables, [u8; 64]), String> {
    use sha2::{Digest, Sha512};
    use serde_json::Value;
    let pin: [u8; 64] = Sha512::digest(text.as_bytes()).into();
    let mut lines = text.split_inclusive('\n');
    lines.next().and_then(|h| h.trim_end().strip_prefix("flock-tables ")).ok_or("not a flock-tables file")?;
    let meta: Value = serde_json::from_str(lines.next().and_then(|l| l.trim_end().strip_prefix("META ")).ok_or("no META line")?)
        .map_err(|e| format!("META: {e}"))?;
    if meta["format"] != FORMAT || meta["statement"] != STATEMENT {
        return Err("META: not verity/flock-tables".into());
    }
    let mut circuits = std::collections::BTreeMap::new();
    while let Some(l) = lines.next() {
        let h: Vec<&str> = l.split_whitespace().collect();
        if h.len() != 3 || h[0] != "CIRCUIT" {
            return Err(format!("tables: expected a CIRCUIT line, found {:?}", l.chars().take(40).collect::<String>()));
        }
        let n: usize = h[2].parse().map_err(|_| "CIRCUIT line count")?;
        let body: String = (0..n).map(|_| lines.next().unwrap_or("")).collect();
        let net = std::panic::catch_unwind(|| crate::ir_block::IrUnitNet::parse(&body)).map_err(|_| format!("CIRCUIT {}: not a flock-ir-unit/v2 circuit", h[1]))?;
        if circuits.insert(h[1].to_string(), net).is_some() {
            return Err(format!("CIRCUIT {} twice", h[1]));
        }
    }
    let tabs = meta["tables"].as_array().ok_or("META: tables")?;
    let names: Vec<&str> = tabs.iter().map(|t| t["name"].as_str().unwrap_or("")).collect();
    let (mut templates, mut counts) = (Vec::new(), Vec::new());
    for t in tabs {
        let c = t["circuit"].as_str().and_then(|c| circuits.get(c)).ok_or("META: a table's circuit is not a CIRCUIT section")?;
        templates.push(Template::from_net(c));
        counts.push(t["blocks"].as_u64().filter(|&n| n > 0).ok_or("META: a table's blocks")? as usize);
    }
    let region = |r: &Value| -> Result<TableRegion, String> {
        let table = names.iter().position(|&n| Some(n) == r["table"].as_str()).ok_or("glue: no such table")?;
        let fixed = r["fixed"].as_object().ok_or("glue: fixed")?.iter()
            .map(|(k, v)| Ok((k.parse::<usize>().map_err(|_| "glue: a fixed bit")?, v.as_u64().filter(|&b| b < 2).ok_or("glue: a fixed value")? == 1)))
            .collect::<Result<Vec<_>, String>>()?;
        let free = r["free"].as_array().ok_or("glue: free")?.iter().map(|b| b.as_u64().map(|b| b as usize).ok_or("glue: a free bit"))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(TableRegion { table, fixed, free })
    };
    let list = |v: &Value| -> Result<Vec<usize>, String> {
        v.as_array().ok_or("glue: a map's list")?.iter().map(|x| x.as_u64().map(|x| x as usize).ok_or_else(|| "glue: a map entry".to_string())).collect()
    };
    let glue = meta["glue"].as_array().ok_or("META: glue")?.iter().map(|g| {
        let map = match &g["map"] {
            Value::String(s) if s == "id" => Map::Id,
            m if m.get("perm").is_some() => Map::Perm(list(&m["perm"])?),
            m if m.get("broadcast").is_some() => Map::Broadcast(list(&m["broadcast"])?),
            _ => return Err("glue: a map is id, perm or broadcast".to_string()),
        };
        Ok(Glue { dst: region(&g["dst"])?, src: region(&g["src"])?, map })
    }).collect::<Result<Vec<_>, String>>()?;
    Ok((Tables::new(templates, counts, glue)?, pin))
}

impl Tables {
    /// Load-time checks: slot order, counts within the capacity, every region inside its table, every destination an input
    /// row, no destination twice, and the relations well formed.
    pub fn new(templates: Vec<Template>, counts: Vec<usize>, glue: Vec<Glue>) -> Result<Self, String> {
        if templates.windows(2).any(|w| w[0].k_log < w[1].k_log) || templates.len() != counts.len() {
            return Err("tables: one count per table, in k_log-descending order".into());
        }
        let nu = counts.iter().map(|&n| n.max(8).next_power_of_two().trailing_zeros() as usize).max().unwrap_or(3);
        let registry = Registry::new(templates.iter().map(Template::table_type).collect(), nu);
        let circuits = templates.iter().map(|t| {
            let tt = t.table_type();
            SparseMatrixCircuit::new(leak(tt.a_0), leak(tt.b_0)).with_const_pin(Some(t.const_pos))
        }).collect();
        let mut s = Tables { templates, counts, glue, registry, nu, circuits, rels: Vec::new(), glue_s: Default::default() };
        let m = s.registry.m_total();
        let mut dsts = std::collections::BTreeSet::new();
        let mut rels = Vec::new();
        for g in &s.glue {
            let (dst, src) = (s.region(&g.dst)?, s.region(&g.src)?);
            let t = &s.templates[g.dst.table];
            for x in 0..1usize << g.dst.free.len() {
                let local: usize = g.dst.fixed.iter().map(|&(b, v)| (v as usize) << b).sum::<usize>()
                    + g.dst.free.iter().enumerate().map(|(j, &b)| (x >> j & 1) << b).sum::<usize>();
                if local & ((1 << t.k_log) - 1) >= t.n_in {
                    return Err(format!("glue: destination row {} of table {} is not an input row", local & ((1 << t.k_log) - 1), g.dst.table));
                }
                if !dsts.insert((g.dst.table, local)) {
                    return Err("glue: a destination position is glued twice".into());
                }
            }
            rels.push(Relation { dst, src, map: g.map.clone() });
        }
        glue::check(&rels, m)?;
        s.rels = rels;
        Ok(s)
    }

    /// A table's region as a sub-cube of the union's address space.
    fn region(&self, r: &TableRegion) -> Result<Region, String> {
        let t = self.templates.get(r.table).ok_or("glue: no such table")?;
        let slot = self.registry.slots()[r.table];
        let (k, nu, m) = (t.k_log, self.nu, self.registry.m_total());
        let b = self.counts[r.table].max(1).next_power_of_two().trailing_zeros() as usize;
        let addr = |l: usize| if l < 7 { l } else if l < k { l + nu } else { 7 + l - k };
        let mut local_seen = vec![false; k + b];
        let (mut mask, mut fixed) = (0u64, 0u64);
        for &(l, v) in &r.fixed {
            if l >= k + b || std::mem::replace(&mut local_seen[l], true) {
                return Err("glue: a region's fixed bit is outside its table or repeated".into());
            }
            mask |= 1 << addr(l);
            fixed |= (v as u64) << addr(l);
        }
        let mut free = Vec::with_capacity(r.free.len());
        for &l in &r.free {
            if l >= k + b || std::mem::replace(&mut local_seen[l], true) {
                return Err("glue: a region's free bit is outside its table or repeated".into());
            }
            free.push(addr(l));
        }
        if local_seen.iter().any(|&x| !x) {
            return Err("glue: a region leaves a table bit neither fixed nor free".into());
        }
        for l in k + b..k + nu {
            mask |= 1 << addr(l);
        }
        for a in slot.m_slot..m {
            mask |= 1 << a;
            fixed |= ((slot.offset >> a) as u64 & 1) << a;
        }
        Ok(Region { mask, fixed, free })
    }

    pub fn union(&self) -> UnionInstance<'_> {
        UnionInstance::new(&self.registry, self.counts.clone())
    }

    pub fn params(&self, profile: flock_core::pcs::ligerito::LigeritoProfile) -> PcsParams {
        let u = self.union();
        let m = u.dense_m();
        let batch = flock_core::pcs::ligerito::embedded_initial_k_or_default(m, profile);
        #[cfg(feature = "sha512")]
        let merkle_hash = flock_core::merkle::HashKind::Sha512;
        #[cfg(not(feature = "sha512"))]
        let merkle_hash = Default::default();
        PcsParams { m, log_inv_rate: profile.log_inv_rate(), log_batch_size: batch, profile, num_lanes: u.commit_lanes(batch), merkle_hash }
    }

    /// Every block's `(z, a, b)` from its inputs (`inputs[t][j]`), batch-major per slot, with its lincheck stripe.
    fn slot(&self, t: usize, blocks: &[Vec<bool>]) -> (Vec<F128>, Vec<F128>, Vec<F128>, Vec<u8>) {
        let tp = &self.templates[t];
        let (k, nu) = (tp.k_log, self.nu);
        let words = 1usize << (k + nu - 7);
        let (mut z, mut a, mut b) = (vec![F128::ZERO; words], vec![F128::ZERO; words], vec![F128::ZERO; words]);
        let mut logical = vec![false; 1 << (k + nu)];
        let set = |v: &mut [F128], w: usize, bit: usize| if bit < 64 { v[w].lo |= 1 << bit } else { v[w].hi |= 1 << (bit - 64) };
        for (j, ins) in blocks.iter().enumerate() {
            let (zz, aa, bb) = tp.eval(ins);
            for c in 0..1usize << k {
                let w = ((c >> 7) << nu) + j;
                for (v, bits) in [(&mut z, &zz), (&mut a, &aa), (&mut b, &bb)] {
                    if bits[c] {
                        set(v, w, c & 127);
                    }
                }
                logical[(j << k) + c] = zz[c];
            }
        }
        (z, a, b, pack_z_lincheck(&logical, k + nu, k))
    }

    /// Table `t`'s block `j`, in-block bit `c`: its address in the union's witness.
    pub fn address(&self, t: usize, j: usize, c: usize) -> usize {
        self.registry.slots()[t].offset + ((((c >> 7) << self.nu) + j) << 7) + (c & 127)
    }

    /// The honest statement's proof from every table's blocks' inputs (`inputs[t][j]`, `counts[t]` blocks each).
    pub fn prove<Ch: Challenger>(&self, inputs: &[Vec<Vec<bool>>], profile: flock_core::pcs::ligerito::LigeritoProfile, ch: &mut Ch)
        -> (Commitment, R1csProofMergedLigerito, GlueProof) {
        self.prove_glue_over(inputs, profile, &|_| {}, ch)
    }

    /// [`Tables::prove`] with the glue sumcheck run over the committed witness as `alter` changes it (a cheating prover).
    fn prove_glue_over<Ch: Challenger>(&self, inputs: &[Vec<Vec<bool>>], profile: flock_core::pcs::ligerito::LigeritoProfile,
                                       alter: &dyn Fn(&mut Vec<F128>), ch: &mut Ch) -> (Commitment, R1csProofMergedLigerito, GlueProof) {
        let slots: Vec<UnionSlotProverInput<'_>> = (0..self.templates.len())
            .map(|t| UnionSlotProverInput::new(self.slot(t, &inputs[t]), &self.circuits[t] as &dyn LincheckCircuit))
            .collect();
        let m = self.registry.m_total();
        let mut gp = None;
        let (proof, commitment, _) = prove_fast_ligerito_union_glued(&self.union(), &self.params(profile), slots, &mut |_cm: &Commitment, z: &[F128], ch: &mut Ch| {
            let t = std::time::Instant::now();
            let mut z = z.to_vec();
            alter(&mut z);
            let (p, rho) = glue::prove_packed(&self.rels, m, &z, ch);
            self.glue_s.set((t.elapsed().as_secs_f64(), self.glue_s.get().1));
            let claim = LinkClaim { point: rho, value: p.value };
            gp = Some(p);
            vec![claim]
        }, ch);
        (commitment, proof, gp.expect("the glue hook ran"))
    }

    pub fn verify<Ch: Challenger>(&self, commitment: &Commitment, proof: &R1csProofMergedLigerito, gp: &GlueProof,
                                  profile: flock_core::pcs::ligerito::LigeritoProfile, ch: &mut Ch) -> Result<(), FlockVerifyError> {
        let circs: Vec<&dyn LincheckCircuit> = self.circuits.iter().map(|c| c as &dyn LincheckCircuit).collect();
        let m = self.registry.m_total();
        verify_ligerito_union_glued(&self.union(), &circs, commitment, proof, &self.params(profile), &mut |_cm: &Commitment, ch: &mut Ch| {
            let t = std::time::Instant::now();
            let got = glue::verify(&self.rels, m, gp, ch).map(|(point, value)| vec![LinkClaim { point, value }]);
            self.glue_s.set((self.glue_s.get().0, t.elapsed().as_secs_f64()));
            got
        }, ch).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use flock_core::{challenger::FsChallenger, pcs::ligerito::LigeritoProfile};

    use super::*;

    const K: usize = 16;
    const BLOCKS: usize = 64;

    /// A `2^16`-row template (big enough for a registered Ligerito shape): inputs rows 0..31, the constant at 32, row `64 + j` =
    /// `x_j AND x_{j+s}` (j < 32), then filler gates to the end.
    fn template(s: usize) -> Template {
        let k = 1 << K;
        let (mut a, mut b) = (vec![Vec::new(); k], vec![Vec::new(); k]);
        for j in 0..32 {
            a[64 + j] = vec![j];
            b[64 + j] = vec![(j + s) % 32];
        }
        for i in 96..k {
            a[i] = vec![i - 1, i % 32];
            b[i] = vec![i - 2, 32];
        }
        Template { k_log: K, n_in: 32, const_pos: 32, useful: k, a, b }
    }

    /// Rows `base..base+32` of every block: in-block bits 0..4 and the block bits free, in-block bits 5..15 fixed to `base`.
    fn rows(t: usize, base: usize) -> TableRegion {
        TableRegion { table: t, fixed: (5..K).map(|l| (l, base >> l & 1 == 1)).collect(), free: (0..5).chain(K..K + 6).collect() }
    }

    /// Table 0 produces rows 64..95 of its 64 blocks; table 1 reads them as its inputs 0..31 (identity over rows and blocks).
    fn statement() -> Tables {
        let glue = vec![Glue { dst: rows(1, 0), src: rows(0, 64), map: Map::Id }];
        Tables::new(vec![template(1), template(3)], vec![BLOCKS, BLOCKS], glue).unwrap()
    }

    fn inputs(flip: Option<(usize, usize)>) -> Vec<Vec<Vec<bool>>> {
        let t0 = template(1);
        let mut s = 0x2545f4914f6cdd1du64;
        let p: Vec<Vec<bool>> = (0..BLOCKS).map(|_| (0..32).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s & 1 == 1 }).collect()).collect();
        let mut c: Vec<Vec<bool>> = p.iter().map(|ins| t0.eval(ins).0[64..96].to_vec()).collect();
        if let Some((j, i)) = flip {
            c[j][i] = !c[j][i];
        }
        vec![p, c]
    }

    fn run(st: &Tables, ins: &[Vec<Vec<bool>>]) -> Result<(), FlockVerifyError> {
        let (cm, proof, gp) = st.prove(ins, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test"));
        st.verify(&cm, &proof, &gp, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test"))
    }

    /// A `flock-ir-unit/v2` text: 32 input bits (word 0), 32 output bits at word 1 (row `128 + j` = `x_j AND x_{j+s}`), filler
    /// gates to `2^16` rows, the constant last.
    fn net_text(name: &str, s: usize) -> String {
        let useful = 1 << K;
        let c = useful - 1;
        let mut t = format!("flock-ir-unit/v2 {name} {useful} {c} 1 0 1 1 32 1 1 1 1 32\n");
        for i in 0..useful {
            let row = if i < 32 || i == c {
                format!("1 {i} 1 {i}")
            } else if (128..160).contains(&i) {
                format!("1 {} 1 {}", i - 128, (i - 128 + s) % 32)
            } else if i >= 160 {
                format!("1 {} 1 {}", i - 1, i - 2)
            } else {
                "0 0".to_string()
            };
            t.push_str(&row);
            t.push('\n');
        }
        t
    }

    #[test]
    fn a_tables_file_parses_proves_and_pins() {
        let (p, q) = (net_text("producer", 1), net_text("consumer", 3));
        let region = |t: &str, base: usize| serde_json::json!({"table": t, "fixed": (5..K).map(|l| (l.to_string(), serde_json::json!(base >> l & 1))).collect::<serde_json::Map<_, _>>(),
                                                          "free": (0..5).chain(K..K + 6).collect::<Vec<_>>()});
        let meta = serde_json::json!({"format": FORMAT, "statement": STATEMENT,
            "tables": [{"name": "p", "circuit": "producer", "blocks": BLOCKS}, {"name": "c", "circuit": "consumer", "blocks": BLOCKS}],
            "glue": [{"dst": region("c", 0), "src": region("p", 128), "map": "id"}]});
        let text = format!("flock-tables demo\nMETA {meta}\nCIRCUIT producer {}\n{p}CIRCUIT consumer {}\n{q}", p.lines().count(), q.lines().count());
        let (st, pin) = parse(&text).unwrap();
        assert_eq!(pin.len(), 64);
        let mut s = 0x9e3779b97f4a7c15u64;
        let ins_p: Vec<Vec<bool>> = (0..BLOCKS).map(|_| (0..128).map(|i| i < 32 && { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s & 1 == 1 }).collect()).collect();
        let ins_c: Vec<Vec<bool>> = ins_p.iter().map(|x| { let z = st.templates[0].eval(x).0; (0..128).map(|i| i < 32 && z[128 + i]).collect() }).collect();
        run(&st, &[ins_p, ins_c]).unwrap();
        let other = text.replace("\"map\":\"id\"", "\"map\":{\"broadcast\":[0,1,2,3,4,5,6,7,8,9,10]}");
        assert_ne!(parse(&other).unwrap().1, pin, "an edited descriptor is another pin");
        let into_output = text.replace(&serde_json::to_string(&region("c", 0)).unwrap(), &serde_json::to_string(&region("c", 128)).unwrap());
        assert!(parse(&into_output).err().unwrap().contains("not an input row"));
    }

    #[test]
    fn gf128_multiplier_matches_carryless_multiplication() {
        let t = Template::gf128_mul();
        // bit i is x^i's coefficient: shift-and-add, reducing x^128 to x^7 + x^2 + x + 1 as it goes
        let mul = |mut a: u128, b: u128| -> u128 {
            let mut c = 0u128;
            for i in 0..128 {
                if b >> i & 1 == 1 {
                    c ^= a;
                }
                a = (a << 1) ^ if a >> 127 == 1 { 0x87 } else { 0 };
            }
            c
        };
        let mut s = 0x9e3779b97f4a7c15u64;
        let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s as u128 };
        for _ in 0..4 {
            let (a, b) = (next() | next() << 64, next() | next() << 64);
            let ins: Vec<bool> = (0..256).map(|i| if i < 128 { a >> i & 1 == 1 } else { b >> (i - 128) & 1 == 1 }).collect();
            let z = t.eval(&ins).0;
            let c = (0..128).fold(0u128, |c, d| c | (z[512 + 16384 + d] as u128) << d);
            assert_eq!(c, mul(a, b));
        }
    }

    /// private-recursion's two-table demo: SHA-512 compressions (the composite circuit's `sha512x3` slot, three chained
    /// compressions per block) feeding a table of GF(2^128) multipliers, `a` and `b` two of the compressions' output words, glued
    /// in `FC_GLUE_SPLIT` relations; prints the prover's and verifier's glue time against the rest. `FC_SHA512_CIRCUIT`: a
    /// staged `circuit.txt`.
    #[test]
    #[ignore]
    fn sha512_feeding_gf128_multipliers() {
        let Ok(path) = std::env::var("FC_SHA512_CIRCUIT") else { return };
        let split: usize = std::env::var("FC_GLUE_SPLIT").ok().and_then(|x| x.parse().ok()).unwrap_or(1);
        let text = std::fs::read_to_string(path).unwrap();
        let c = crate::circuit::Composite::parse(&text, None).unwrap();
        let sha = Template::from_net(&c.nets[c.sha_net].1);
        let out = c.nets[c.sha_net].1.out_groups.last().unwrap().col;
        let (k, blocks) = (sha.k_log, 8);
        let rows = |t: usize, kk: usize, base: usize, width: usize| TableRegion {
            table: t,
            fixed: (width.trailing_zeros() as usize..kk).map(|l| (l, base >> l & 1 == 1)).collect(),
            free: (0..width.trailing_zeros() as usize).chain(kk..kk + 3).collect(),
        };
        let w = 128 / split;
        let glue: Vec<Glue> = (0..2).flat_map(|h| (0..split).map(move |q| (h, q)))
            .map(|(h, q)| Glue { dst: rows(1, 15, 128 * h + w * q, w), src: rows(0, k, 128 * (out + h) + w * q, w), map: Map::Id }).collect();
        let st = Tables::new(vec![sha.clone(), Template::gf128_mul()], vec![blocks, blocks], glue).unwrap();
        let mut s = 0x2545f4914f6cdd1du64;
        let sha_in: Vec<Vec<bool>> = (0..blocks).map(|_| (0..sha.n_in).map(|_| { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s & 1 == 1 }).collect()).collect();
        let mul_in: Vec<Vec<bool>> = sha_in.iter().map(|x| sha.eval(x).0[128 * out..128 * out + 256].to_vec()).collect();
        let ins = vec![sha_in, mul_in];
        let t = std::time::Instant::now();
        let (cm, proof, gp) = st.prove(&ins, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-demo"));
        let prove_s = t.elapsed().as_secs_f64();
        let t = std::time::Instant::now();
        st.verify(&cm, &proof, &gp, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-demo")).unwrap();
        let verify_s = t.elapsed().as_secs_f64();
        let (gp_s, gv_s) = st.glue_s.get();
        println!("DEMO relations={} M={} dense_m={} prove_s={prove_s:.3} glue_prove_s={gp_s:.3} verify_s={verify_s:.4} glue_verify_s={gv_s:.5}",
                 st.rels.len(), st.registry.m_total(), st.union().dense_m());
    }

    #[test]
    fn two_tables_with_private_glue_prove_and_verify() {
        let st = statement();
        run(&st, &inputs(None)).unwrap();
    }

    #[test]
    fn a_glued_bit_that_differs_is_refused_by_the_glue() {
        // each table's witness is honest for its own inputs; only the glue is broken
        let st = statement();
        assert_eq!(run(&st, &inputs(Some((2, 17)))), Err(FlockVerifyError::Glue));
    }

    #[test]
    fn a_glue_claim_off_the_committed_witness_is_refused_by_the_opening() {
        // the committed destination bit differs; the prover runs the glue on a copy with it repaired, so the sumcheck holds
        // and only the merged opening of z~(rho) can catch it
        let st = statement();
        let (j, i) = (2, 17);
        let at = st.address(1, j, i);
        let repair = move |z: &mut Vec<F128>| {
            let (w, t) = (at >> 7, at & 127);
            if t < 64 { z[w].lo ^= 1 << t } else { z[w].hi ^= 1 << (t - 64) }
        };
        let (cm, proof, gp) = st.prove_glue_over(&inputs(Some((j, i))), LigeritoProfile::Fast100, &repair, &mut FsChallenger::new(b"tables-test"));
        let got = st.verify(&cm, &proof, &gp, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test"));
        assert!(matches!(got, Err(FlockVerifyError::PcsOpen(_))), "{got:?}");
    }

    #[test]
    fn an_edited_glue_descriptor_is_refused() {
        let st = statement();
        let ins = inputs(None);
        let (cm, proof, gp) = st.prove(&ins, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test"));
        let mut other = statement();
        other.rels[0].map = Map::Perm((0..11).map(|j| if j < 2 { 1 - j } else { j }).collect());
        assert!(other.verify(&cm, &proof, &gp, LigeritoProfile::Fast100, &mut FsChallenger::new(b"tables-test")).is_err());
    }

    #[test]
    fn a_destination_that_is_not_an_input_row_is_refused_at_load() {
        let glue = vec![Glue { dst: rows(1, 64), src: rows(0, 64), map: Map::Id }];
        let err = Tables::new(vec![template(1), template(3)], vec![BLOCKS, BLOCKS], glue).err().unwrap();
        assert!(err.contains("not an input row"), "{err}");
    }
}
