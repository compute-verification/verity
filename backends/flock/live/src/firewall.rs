//! The firewall's send check on the outer `--zk` prover (PROTOCOL.md §10.1): every live round the device frames is held to
//! the statement's skeleton in the algebraic phase, and byte for byte to an opening the firewall computes itself (the shadow,
//! in lock-step on the same coins) in the opening phase, before the masking challenger sends any of it. A refusal is a
//! `FIREWALL-REFUSED: <what>` message, raised before anything that depends on the refused round leaves.

use std::sync::mpsc::{Receiver, Sender, channel};

use flock_field::F128;
use flock_transcript::challenger::Challenger;

use crate::{KIND_SCALAR, KIND_SLICE, OP_BYTES, OP_DOMAIN, OP_LABEL, OP_OBSERVE, OP_SQUEEZE, push_header, push_padded};

pub fn refused(what: impl std::fmt::Display) -> String {
    format!("FIREWALL-REFUSED: {what}")
}

/// One framed item of a round, as the transcript absorbs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Item {
    Domain(Vec<u8>),
    Label(Vec<u8>),
    Bytes(Vec<u8>),
    /// an observe: its kind and the values (the skeleton's: each value's index among the exposed values)
    Obs { kind: u8, vals: Vec<F128> },
    Squeeze { kind: u8, n: usize },
}

impl std::fmt::Display for Item {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = |b: &[u8]| if b.iter().all(|c| c.is_ascii_graphic()) { format!("{:?}", String::from_utf8_lossy(b)) } else { format!("{} bytes", b.len()) };
        match self {
            Item::Domain(b) => write!(f, "domain {}", text(b)),
            Item::Label(b) => write!(f, "label {}", text(b)),
            Item::Bytes(b) => write!(f, "bytes ({})", b.len()),
            Item::Obs { kind, vals } => write!(f, "observe {} of {}", if *kind == KIND_SCALAR { "scalar" } else { "slice" }, vals.len()),
            Item::Squeeze { kind, n } => write!(f, "squeeze {} of {n}", if *kind == KIND_SCALAR { "scalar" } else { "slice" }),
        }
    }
}

/// A round's items (transcript-v2 frames: 16-byte header `[op][kind][0;6][len u64 LE]`, payload zero-padded to 16).
pub fn parse(content: &[u8]) -> Result<Vec<Item>, String> {
    let body = |at: usize, w: usize| content.get(at..at + w).ok_or_else(|| refused("a frame overruns its round"));
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < content.len() {
        let h = body(at, 16)?;
        let (op, kind, len) = (h[0], h[1], u64::from_le_bytes(h[8..].try_into().unwrap()) as usize);
        if h[2..8].iter().any(|b| *b != 0) {
            return Err(refused("a frame header's reserved bytes are not zero"));
        }
        at += 16;
        let padded = len.div_ceil(16) * 16;
        let payload = |at: usize| -> Result<Vec<u8>, String> {
            let p = body(at, padded)?;
            if p[len..].iter().any(|b| *b != 0) {
                return Err(refused("a frame's padding is not zero"));
            }
            Ok(p[..len].to_vec())
        };
        match op {
            OP_DOMAIN | OP_LABEL | OP_BYTES if kind == 0 => {
                let p = payload(at)?;
                out.push(match op { OP_DOMAIN => Item::Domain(p), OP_LABEL => Item::Label(p), _ => Item::Bytes(p) });
                at += padded;
            }
            OP_OBSERVE if kind == KIND_SCALAR || kind == KIND_SLICE => {
                let vals = body(at, 16 * len)?.chunks_exact(16)
                    .map(|c| F128::new(u64::from_le_bytes(c[..8].try_into().unwrap()), u64::from_le_bytes(c[8..].try_into().unwrap()))).collect();
                out.push(Item::Obs { kind, vals });
                at += 16 * len;
            }
            OP_SQUEEZE if kind == KIND_SCALAR || kind == KIND_SLICE => out.push(Item::Squeeze { kind, n: len }),
            _ => return Err(refused(format!("a frame of op {op} and kind {kind}"))),
        }
    }
    Ok(out)
}

/// The index each skeleton value stands for: [`tagged`]'s values carry it in their low word.
const TAG: u64 = 0x6761_7465_7761_7921;

/// `n` stand-in exposed values, value `i` carrying its index `i`, for [`Recorder`] to read back.
pub fn tagged(n: usize) -> Vec<F128> {
    (0..n as u64).map(|i| F128::new(i, TAG)).collect()
}

/// A challenger that records every call a replay makes, as the device's live challenger frames it (observe kinds, and nonce 0
/// absorbed as bytes), answering distinct coins that are neither 0 nor 1 (the replay divides by `1 + r`).
#[derive(Default)]
pub struct Recorder {
    pub items: Vec<Item>,
    coins: u64,
}

impl Recorder {
    fn coin(&mut self) -> F128 {
        self.coins += 1;
        F128::new(0x5eed_0000 + self.coins, 0x1234_5678_9abc_def0)
    }
}

impl Challenger for Recorder {
    fn observe_label(&mut self, l: &[u8]) {
        self.items.push(Item::Label(l.to_vec()));
    }
    fn observe_f128(&mut self, v: F128) {
        self.items.push(Item::Obs { kind: KIND_SCALAR, vals: vec![v] });
    }
    fn observe_f128_slice(&mut self, vs: &[F128]) {
        self.items.push(Item::Obs { kind: KIND_SLICE, vals: vs.to_vec() });
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        self.items.push(Item::Bytes(b.to_vec()));
    }
    fn sample_f128(&mut self) -> F128 {
        self.items.push(Item::Squeeze { kind: KIND_SCALAR, n: 1 });
        self.coin()
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.items.push(Item::Squeeze { kind: KIND_SLICE, n });
        (0..n).map(|_| self.coin()).collect()
    }
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        self.observe_bytes(&0u64.to_le_bytes());
        0
    }
    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!("the algebraic phase does not fork")
    }
}

/// The algebraic phase's items as the device frames them, from the statement's start: the domain, `bind_statement`'s label,
/// digest and cap, then what `replay` (the verifier's replay of the masked zerocheck and lincheck, run on [`tagged`] values
/// into a [`Recorder`]) absorbs and draws, without the pads' label and root that the masking challenger inserts itself.
pub fn skeleton(domain: &[u8], label_r1cs: &[u8], digest: &[u8], cap: &[u8], label_zc: &[u8], label_pads: &[u8],
                replay: impl FnOnce(&mut Recorder) -> Result<(), String>) -> Result<Vec<Item>, String> {
    let mut r = Recorder::default();
    replay(&mut r)?;
    let at = r.items.iter().position(|x| *x == Item::Label(label_zc.to_vec())).ok_or("the replay absorbs no zerocheck label")?;
    match &r.items[at + 1..at + 3] {
        [Item::Label(l), Item::Bytes(_)] if l == label_pads => {}
        _ => return Err("the replay's pads label and root do not follow the zerocheck label".into()),
    }
    r.items.drain(at + 1..at + 3);
    let mut out = vec![Item::Domain(domain.to_vec()), Item::Label(label_r1cs.to_vec()), Item::Bytes(digest.to_vec()), Item::Bytes(cap.to_vec())];
    out.extend(r.items);
    Ok(out)
}

/// The algebraic phase's check: each round's items against the skeleton, one by one. Labels, bytes and the domain must equal
/// the skeleton's; an observe must have its kind and length (its values are masked downstream and are logged here, at the
/// exposed index the skeleton gives); a squeeze its kind and coin count.
pub struct Checker {
    items: Vec<Item>,
    at: usize,
    round: usize,
    pub values: Vec<Option<F128>>,
}

impl Checker {
    pub fn new(items: Vec<Item>, n_values: usize) -> Self {
        Checker { items, at: 0, round: 0, values: vec![None; n_values] }
    }

    /// Whether every item of the algebraic phase has passed (the next round is the opening's).
    pub fn done(&self) -> bool {
        self.at == self.items.len()
    }

    pub fn round(&mut self, content: &[u8], n: usize) -> Result<(), String> {
        let got = parse(content)?;
        let r = self.round;
        self.round += 1;
        for (k, g) in got.iter().enumerate() {
            let want = self.items.get(self.at).ok_or_else(|| refused(format!("algebraic phase, round {r}: {g} past the phase's end")))?;
            let at = |what: String| refused(format!("algebraic phase, round {r}, frame {k}: {what}"));
            match (g, want) {
                (Item::Obs { kind, vals }, Item::Obs { kind: wk, vals: idx }) if kind == wk && vals.len() == idx.len() => {
                    for (v, i) in vals.iter().zip(idx) {
                        if i.hi != TAG || i.lo as usize >= self.values.len() {
                            return Err(at("the skeleton observes a value that is not an exposed one".into()));
                        }
                        self.values[i.lo as usize] = Some(*v);
                    }
                }
                (Item::Obs { .. }, _) => return Err(at(format!("{g} where the statement has {want}"))),
                (Item::Squeeze { .. }, _) | (Item::Label(_), _) | (Item::Bytes(_), _) | (Item::Domain(_), _) if g != want => {
                    return Err(at(format!("{g} where the statement has {want}")));
                }
                _ => {}
            }
            self.at += 1;
        }
        match got.last() {
            Some(Item::Squeeze { n: k, .. }) if *k == n && n > 0 => Ok(()),
            _ => Err(refused(format!("algebraic phase, round {r}: the round does not end with its squeeze of {n}"))),
        }
    }
}

/// The firewall's end of the shadow opening's lock-step: each of the shadow's rounds, to compare with the device's, and the coins
/// back. The shadow ([`ShadowEnd`] in a [`Framer`]) runs on its own thread.
pub struct ShadowLink {
    rx: Receiver<(Vec<u8>, usize)>,
    tx: Option<Sender<Vec<F128>>>,
    pub round: usize,
}

pub struct ShadowEnd {
    tx: Sender<(Vec<u8>, usize)>,
    rx: Receiver<Vec<F128>>,
}

pub fn shadow_pair() -> (ShadowLink, ShadowEnd) {
    let (tx_r, rx_r) = channel();
    let (tx_c, rx_c) = channel();
    (ShadowLink { rx: rx_r, tx: Some(tx_c), round: 0 }, ShadowEnd { tx: tx_r, rx: rx_c })
}

impl ShadowEnd {
    /// The shadow's round: hand it to the firewall and wait for its coins (the shadow stops when the firewall has gone).
    pub fn round(&mut self, content: &[u8], n: usize) -> Vec<F128> {
        self.tx.send((content.to_vec(), n)).ok().and_then(|_| self.rx.recv().ok())
            .unwrap_or_else(|| panic!("the firewall stopped the shadow opening"))
    }
}

impl ShadowLink {
    /// The device's opening round against the shadow's next one: refused unless the two are byte for byte the same.
    pub fn check(&mut self, content: &[u8], n: usize) -> Result<(), String> {
        let r = self.round;
        self.round += 1;
        let (want, k) = self.rx.recv().map_err(|_| refused(format!("opening round {r}: the firewall's opening ended before the device's")))?;
        #[cfg(feature = "sha512")]
        crate::firewall_transcript::with(|t| t.shadow_round(&want, k));
        if want == content && k == n {
            return Ok(());
        }
        let (g, w) = (parse(content)?, parse(&want).map_err(|e| format!("the shadow's round: {e}"))?);
        let show = |x: Option<&Item>| x.map_or("nothing".into(), |x| x.to_string());
        match (0..g.len().max(w.len())).find(|&i| g.get(i) != w.get(i)) {
            Some(i) if show(g.get(i)) == show(w.get(i)) => {
                Err(refused(format!("opening round {r}, frame {i}: the device's {} differs from the firewall's in its payload", show(g.get(i)))))
            }
            Some(i) => Err(refused(format!("opening round {r}, frame {i}: the device's {} is not the firewall's {}", show(g.get(i)), show(w.get(i))))),
            None => Err(refused(format!("opening round {r}: the device squeezes {n} coins, the firewall {k}"))),
        }
    }

    pub fn answer(&mut self, coins: &[F128]) {
        if let Some(t) = &self.tx {
            let _ = t.send(coins.to_vec());
        }
    }

    /// Whether the shadow has a round the device never sent (checked once the device has returned).
    pub fn pending(&self) -> bool {
        self.rx.try_recv().is_ok()
    }

    /// Stop the shadow: its next round finds no firewall.
    pub fn close(&mut self) {
        self.tx = None;
    }
}

/// Frames every call as the device prover's challenger does (`FsChallenger` under its live hook) and hands each round to
/// `rounds` at its squeeze: M1's CPU prover takes the device path's host side through it (`FC_ZK_FRAMED`), and the firewall's
/// shadow opening frames its rounds through it for the comparison.
pub struct Framer<'a> {
    pub pending: Vec<u8>,
    pub rounds: &'a mut (dyn FnMut(&[u8], usize) -> Vec<F128> + Send),
}

impl Framer<'_> {
    fn squeeze(&mut self, kind: u8, n: usize) -> Vec<F128> {
        push_header(&mut self.pending, OP_SQUEEZE, kind, n as u64);
        if n == 0 {
            return Vec::new();
        }
        let c = std::mem::take(&mut self.pending);
        (self.rounds)(&c, n)
    }
}

impl Challenger for Framer<'_> {
    fn observe_label(&mut self, l: &[u8]) {
        push_header(&mut self.pending, OP_LABEL, 0, l.len() as u64);
        push_padded(&mut self.pending, l);
    }
    fn observe_f128(&mut self, v: F128) {
        push_header(&mut self.pending, OP_OBSERVE, KIND_SCALAR, 1);
        self.pending.extend_from_slice(&v.lo.to_le_bytes());
        self.pending.extend_from_slice(&v.hi.to_le_bytes());
    }
    fn observe_f128_slice(&mut self, vs: &[F128]) {
        push_header(&mut self.pending, OP_OBSERVE, KIND_SLICE, vs.len() as u64);
        for v in vs {
            self.pending.extend_from_slice(&v.lo.to_le_bytes());
            self.pending.extend_from_slice(&v.hi.to_le_bytes());
        }
    }
    fn observe_bytes(&mut self, b: &[u8]) {
        push_header(&mut self.pending, OP_BYTES, 0, b.len() as u64);
        push_padded(&mut self.pending, b);
    }
    fn sample_f128(&mut self) -> F128 {
        self.squeeze(KIND_SCALAR, 1)[0]
    }
    fn sample_f128_vec(&mut self, n: usize) -> Vec<F128> {
        self.squeeze(KIND_SLICE, n)
    }
    /// the device's live grinding: nonce 0, absorbed like any message
    fn grind_pow(&mut self, _bits: u32) -> u64 {
        self.observe_bytes(&0u64.to_le_bytes());
        0
    }
    fn verify_pow(&mut self, _: u64, _: u32) -> bool {
        unreachable!("the prover's challenger never verifies")
    }
    fn fork_from_seed(&self, _: [F128; 2], _: &'static [u8]) -> Self {
        unreachable!("the single-table prove path does not fork")
    }
}

/// This process's CPU seconds so far (user and system, every thread), from `/proc/self/stat` in clock ticks of 1/100 s (Linux's
/// `USER_HZ`); 0 where there is no procfs.
pub fn cpu_seconds() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/self/stat") else { return 0.0 };
    let f: Vec<&str> = s.rsplit_once(')').map_or("", |x| x.1).split_whitespace().collect();
    let tick = |i: usize| f.get(i).and_then(|x| x.parse::<f64>().ok()).unwrap_or(0.0);
    (tick(11) + tick(12)) / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn framed(f: impl FnOnce(&mut Framer<'_>)) -> Vec<(Vec<u8>, usize)> {
        let mut out = Vec::new();
        let mut rounds = |c: &[u8], n: usize| { out.push((c.to_vec(), n)); vec![F128::new(7, 7); n] };
        let mut fr = Framer { pending: crate::frame_domain(b"dom"), rounds: &mut rounds };
        f(&mut fr);
        out
    }

    fn phase(ch: &mut impl Challenger, vals: &[F128]) {
        ch.observe_label(b"flock-r1cs-v0");
        ch.observe_bytes(&[1; 64]);
        ch.observe_bytes(&[2; 64]);
        ch.observe_label(b"zc");
        ch.sample_f128_vec(3);
        ch.observe_f128_slice(&vals[..2]);
        ch.observe_f128(vals[2]);
        ch.sample_f128();
    }

    fn checker() -> Checker {
        let mut r = Recorder::default();
        phase(&mut r, &tagged(3));
        let mut items = vec![Item::Domain(b"dom".to_vec())];
        items.extend(r.items);
        Checker::new(items, 3)
    }

    #[test]
    fn checker_logs_values_and_refuses_another_skeleton() {
        let v = [F128::new(10, 0), F128::new(11, 0), F128::new(12, 0)];
        let rounds = framed(|f| phase(f, &v));
        let mut c = checker();
        for (r, n) in &rounds {
            c.round(r, *n).unwrap();
        }
        assert!(c.done());
        assert_eq!(c.values, v.map(Some).to_vec());
        let bad = framed(|f| {
            f.observe_label(b"flock-r1cs-v1");
            f.sample_f128();
        });
        let e = checker().round(&bad[0].0, bad[0].1).unwrap_err();
        assert!(e.starts_with("FIREWALL-REFUSED: algebraic phase, round 0, frame 1: label"), "{e}");
        let short = framed(|f| {
            f.observe_label(b"flock-r1cs-v0");
            f.observe_bytes(&[1; 64]);
            f.observe_bytes(&[2; 64]);
            f.observe_label(b"zc");
            f.sample_f128_vec(2);
        });
        let e = checker().round(&short[0].0, short[0].1).unwrap_err();
        assert!(e.contains("squeeze slice of 2 where the statement has squeeze slice of 3"), "{e}");
    }

    #[test]
    fn shadow_link_names_the_first_differing_frame() {
        let (mut link, mut end) = shadow_pair();
        let t = std::thread::spawn(move || {
            let mut rounds = |c: &[u8], n: usize| end.round(c, n);
            let mut fr = Framer { pending: Vec::new(), rounds: &mut rounds };
            fr.observe_bytes(&[3; 64]);
            fr.sample_f128();
            fr.observe_f128(F128::new(5, 0));
            fr.sample_f128();
        });
        let dev = framed(|f| {
            f.pending.clear();
            f.observe_bytes(&[3; 64]);
            f.sample_f128();
            f.observe_f128(F128::new(6, 0));
            f.sample_f128();
        });
        link.check(&dev[0].0, dev[0].1).unwrap();
        link.answer(&[F128::ONE]);
        let e = link.check(&dev[1].0, dev[1].1).unwrap_err();
        assert_eq!(e, "FIREWALL-REFUSED: opening round 1, frame 0: the device's observe scalar of 1 differs from the firewall's in its payload");
        link.close();
        assert!(t.join().is_err(), "the shadow stops when the firewall closes");
    }
}
