//! The circuit files this verifier accepts, per relation: `pins.txt`, compiled in.  A statement directory is checked
//! against the relation its proof is claimed for (`verify --relation R`) before any file in it is parsed.
//!
//! A circuit set is the manifest's `steps` and the SHA-256 of `circuit.txt`, `epilogue.txt` and `chain.txt`: all the
//! transcript preamble absorbs from the statement besides the public words.  `steps` is part of it because the circuit
//! files are uniform in the unit count (the same files chain any number of units), so they alone do not fix K.

use crate::sha256::Sha256;
use std::path::Path;
use std::sync::OnceLock;

const PINS: &str = include_str!("../pins.txt");

/// One approved circuit set: a line of `pins.txt` (digests as lowercase hex; the generator column is not read here).
#[derive(Debug)]
pub struct Pin {
    pub relation: &'static str,
    pub steps: usize,
    pub circuit: &'static str,
    pub epilogue: &'static str,
    pub chain: &'static str,
}

/// What a statement directory holds that a pin covers.  `chain` is `None` for a manifest that is not checker v2,
/// whose chain the verifier takes from fixed column names instead of `chain.txt`; no such set is pinned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Circuits {
    pub steps: usize,
    pub circuit: String,
    pub epilogue: String,
    pub chain: Option<String>,
}

pub fn hex(d: &[u8]) -> String { d.iter().map(|b| format!("{b:02x}")).collect() }

fn is_digest(s: &str) -> bool { s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) }

fn parse(text: &'static str) -> Result<Vec<Pin>, String> {
    let mut out: Vec<Pin> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let t: Vec<&'static str> = line.split_whitespace().collect();
        if t.len() < 6 { return Err(format!("line {}: relation, steps, three sha256 digests and the generator expected", i + 1)); }
        let steps = t[1].parse::<usize>().ok().filter(|&s| s > 0).ok_or_else(|| format!("line {}: steps {}", i + 1, t[1]))?;
        if let Some(d) = t[2..5].iter().find(|d| !is_digest(d)) { return Err(format!("line {}: {d} is not a lowercase sha256 digest", i + 1)); }
        let pin = Pin { relation: t[0], steps, circuit: t[2], epilogue: t[3], chain: t[4] };
        if let Some(q) = out.iter().find(|q| q.same_set(&pin)) { return Err(format!("line {}: the circuit set of a {} line again", i + 1, q.relation)); }
        out.push(pin);
    }
    Ok(out)
}

pub fn pins() -> &'static [Pin] {
    static P: OnceLock<Vec<Pin>> = OnceLock::new();
    P.get_or_init(|| parse(PINS).unwrap_or_else(|e| panic!("pins.txt: {e}")))
}

/// The pinned relations, in `pins.txt` order.
pub fn relations() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for p in pins() { if !out.contains(&p.relation) { out.push(p.relation); } }
    out
}

impl Pin {
    fn same_set(&self, o: &Pin) -> bool { self.steps == o.steps && self.circuit == o.circuit && self.epilogue == o.epilogue && self.chain == o.chain }

    fn matches(&self, c: &Circuits) -> bool {
        self.steps == c.steps && self.circuit == c.circuit && self.epilogue == c.epilogue && c.chain.as_deref() == Some(self.chain)
    }

    /// The components of `c` that are not this pin's.
    fn diff(&self, c: &Circuits) -> Vec<String> {
        let mut out = Vec::new();
        if c.steps != self.steps { out.push(format!("steps {}, pinned {}", c.steps, self.steps)); }
        for (name, got, want) in [("circuit.txt", &c.circuit, self.circuit), ("epilogue.txt", &c.epilogue, self.epilogue)] {
            if got != want { out.push(format!("{name} sha256 {}..., pinned {}...", &got[..16], &want[..16])); }
        }
        match &c.chain {
            None => out.push(format!("no chain.txt (the manifest is not checker v2), pinned {}...", &self.chain[..16])),
            Some(h) if h != self.chain => out.push(format!("chain.txt sha256 {}..., pinned {}...", &h[..16], &self.chain[..16])),
            Some(_) => {}
        }
        out
    }
}

impl Circuits {
    /// The circuit set of the statement directory `root`; the caller reads `steps` and the checker variant from its
    /// manifest (the verifier's own reading, the one the statement is built from).
    pub fn read(root: &Path, steps: usize, v2: bool) -> Result<Circuits, String> {
        let digest = |name: &str| {
            let p = root.join(name);
            std::fs::read(&p).map(|b| hex(&Sha256::digest(&b))).map_err(|e| format!("{}: {e}", p.display()))
        };
        Ok(Circuits { steps, circuit: digest("circuit.txt")?, epilogue: digest("epilogue.txt")?, chain: if v2 { Some(digest("chain.txt")?) } else { None } })
    }

    /// The circuit set a built statement absorbs: its segments' circuit hashes, its chain spec hash and its `steps`.
    pub fn of(circuit: &[u8; 32], epilogue: &[u8; 32], chain: Option<&[u8; 32]>, steps: usize) -> Circuits {
        Circuits { steps, circuit: hex(circuit), epilogue: hex(epilogue), chain: chain.map(|h| hex(h)) }
    }

    /// The relation this circuit set is pinned for, if any.
    pub fn pinned_relation(&self) -> Option<&'static str> { pins().iter().find(|p| p.matches(self)).map(|p| p.relation) }

    pub fn json(&self) -> String {
        let s = |v: Option<&str>| v.map_or("null".to_string(), |v| format!("\"{v}\""));
        format!(
            "{{\"steps\": {}, \"variant\": {}, \"circuit.txt\": \"{}\", \"epilogue.txt\": \"{}\", \"chain.txt\": {}, \"pinned_relation\": {}}}",
            self.steps, s(Some(if self.chain.is_some() { "v2" } else { "v1" })), self.circuit, self.epilogue, s(self.chain.as_deref()), s(self.pinned_relation())
        )
    }
}

/// `Ok` iff `c` is a pinned circuit set of `relation`; the error says what the files are instead.
pub fn check(c: &Circuits, relation: &str) -> Result<(), String> {
    let rows: Vec<&Pin> = pins().iter().filter(|p| p.relation == relation).collect();
    if rows.is_empty() {
        return Err(format!("no circuits are pinned for relation {relation} in this build (pinned: {}; archive/gkr/verifier/pins.txt)", relations().join(", ")));
    }
    if rows.iter().any(|p| p.matches(c)) { return Ok(()); }
    if let Some(other) = c.pinned_relation() {
        return Err(format!("circuit files are the pinned {other} circuits, the proof is claimed for {relation}"));
    }
    let nearest = rows.iter().map(|p| p.diff(c)).min_by_key(|d| d.len()).unwrap();
    Err(format!("circuit files are not a pinned {relation} circuit set ({}); pass --allow-any-circuit to verify against them unpinned", nearest.join("; ")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(p: &Pin) -> Circuits { Circuits { steps: p.steps, circuit: p.circuit.into(), epilogue: p.epilogue.into(), chain: Some(p.chain.into()) } }

    fn flip(h: &str) -> String { format!("{}{}", if h.starts_with('0') { '1' } else { '0' }, &h[1..]) }

    #[test]
    fn pins_parse_and_name_each_circuit_set_once() {
        let ps = pins();
        assert!(relations().contains(&"bf16-ampere") && relations().contains(&"bf16-hopper"), "{:?}", relations());
        for (i, p) in ps.iter().enumerate() {
            assert!(ps[i + 1..].iter().all(|q| !p.same_set(q)));
            assert_eq!(set(p).pinned_relation(), Some(p.relation));
        }
    }

    #[test]
    fn check_accepts_exactly_the_pinned_sets_of_the_claimed_relation() {
        for p in pins() {
            let c = set(p);
            assert_eq!(check(&c, p.relation), Ok(()));
            for r in relations().into_iter().filter(|&r| r != p.relation) {
                let e = check(&c, r).unwrap_err();
                assert!(e.contains(&format!("the pinned {} circuits, the proof is claimed for {r}", p.relation)), "{e}");
            }
            let variants = [
                ("circuit.txt sha256", Circuits { circuit: flip(&c.circuit), ..c.clone() }),
                ("epilogue.txt sha256", Circuits { epilogue: flip(&c.epilogue), ..c.clone() }),
                ("chain.txt sha256", Circuits { chain: Some(flip(p.chain)), ..c.clone() }),
                ("no chain.txt", Circuits { chain: None, ..c.clone() }),
                ("steps", Circuits { steps: c.steps / 2, ..c.clone() }),
            ];
            for (what, v) in variants {
                let e = check(&v, p.relation).unwrap_err();
                assert!(e.contains(&format!("not a pinned {} circuit set", p.relation)) && e.contains(what), "{what}: {e}");
                assert_eq!(v.pinned_relation(), None);
            }
        }
        let e = check(&set(&pins()[0]), "no-such-relation").unwrap_err();
        assert!(e.contains("no circuits are pinned for relation no-such-relation"), "{e}");
    }

    #[test]
    fn malformed_pin_lines_are_refused() {
        let d = "ab".repeat(32);
        let leak = |s: String| -> &'static str { Box::leak(s.into_boxed_str()) };
        assert_eq!(parse(leak(format!("# comment\n\nr 96 {d} {d} {d} python -m gen\n"))).map(|v| v.len()), Ok(1));
        for bad in [
            format!("r 96 {d} {d} {d}"),                                      // no generator
            format!("r 0 {d} {d} {d} gen"),                                   // steps
            format!("r x {d} {d} {d} gen"),
            format!("r 96 {} {d} {d} gen", &d[1..]),                          // a digest of 63 digits
            format!("r 96 {} {d} {d} gen", d.to_uppercase()),
            format!("r 96 {d} {d} {d} gen\ns 96 {d} {d} {d} gen"),            // one set under two relations
        ] {
            assert!(parse(leak(bad.clone())).is_err(), "{bad}");
        }
    }
}
