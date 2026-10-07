#!/usr/bin/env python3
"""Give Flock-CUDA (b684b12) a live-coin path, and teach the flock-128 GPU harness to use it.

Run from the flock checkout AFTER flock-128's g128_patch.py and BEFORE flock-bench's 22-gpu-unit.sh (which copies
tests/gpu_roundtrip.rs into tests/gpu_unit.rs, so the unit harness inherits everything here).

CUDA side:
- challenger.hpp: FsChallenger gets `live_cb/live_ctx`. When set, every absorbed byte (the transcript-v2 framed stream)
  is also buffered, and each squeeze hands the buffer to `live_cb`, which returns the verifier's coins; nothing is
  derived from the hash. grind_pow returns nonce 0 without searching.
- pow_grind.cuh: grind_pow_device does not search either when live.
- prove_ffi.cu: FlockCudaProveParams gains `live_cb, live_ctx` (appended); the prover's one challenger goes live when
  they are set. The device-side challenger (zc_challenger_device.cuh) is not on the prove path at b684b12.

Harness side (tests/gpu_roundtrip.rs): GPU_LIVE=1 runs every timed run as one live session against the verifier
process at LIVE_VERIFIER (verity/ml/flock/live `flock-live serve-gpu`): the reps' squeezes go through
flock_live::LiveChallenger, the proofs go to the verifier, and the verifier's verdict is the result. With GPU_NEG=1
three extra sessions submit rep 0 as rep 1, a flipped yr[0] and a flipped zerocheck round; each must be rejected.
"""
import pathlib
import re

ch = pathlib.Path("cuda-ghash/challenger.hpp")
s = ch.read_text()
if "live_cb" not in s:
    old = "struct FsChallenger {\n    Sha256 hasher;\n"
    assert s.count(old) == 1
    s = s.replace(old, """struct FsChallenger {
    Sha256 hasher;
    // Live coins (flock-live): when live_cb is set, the absorbed stream is buffered and every squeeze asks the
    // verifier for its coins (out: 2 u64 per coin, lo then hi); the hash state is not used for challenges.
    typedef int (*LiveCb)(void* ctx, const uint8_t* content, size_t len, uint32_t n, uint64_t* out);
    LiveCb live_cb = nullptr;
    void* live_ctx = nullptr;
    std::vector<uint8_t> live_buf;
    void upd(const uint8_t* d, size_t n) {
        hasher.update(d, n);
        if (live_cb) live_buf.insert(live_buf.end(), d, d + n);
    }
    void enable_live(LiveCb cb, void* ctx, const uint8_t* domain, size_t dlen) {
        live_cb = cb; live_ctx = ctx; live_buf.clear(); hasher.reset();
        absorb_header(OP_DOMAIN, 0, (uint64_t)dlen);
        absorb_padded(domain, dlen);
    }
""")
    body_start = s.index("struct FsChallenger {")
    body_end = s.index("// ---- Stratified query schedule")
    body = s[body_start:body_end]
    body = body.replace("hasher.update(h, 16);", "upd(h, 16);")
    body = body.replace("        hasher.update(b, n);\n        size_t rem", "        upd(b, n);\n        size_t rem")
    body = body.replace("hasher.update(z, 16 - rem);", "upd(z, 16 - rem);")
    body = body.replace("le64(b + 8, v.hi); hasher.update(b, 16); }", "le64(b + 8, v.hi); upd(b, 16); }")
    old = "    void squeeze_into(uint8_t* out, size_t n) {\n"
    assert body.count(old) == 1
    body = body.replace(old, old + """        if (live_cb) {
            std::vector<uint64_t> w(n / 8);
            if (n % 16 != 0 || live_cb(live_ctx, live_buf.data(), live_buf.size(), (uint32_t)(n / 16), w.data()) != 0) {
                fprintf(stderr, "flock-live: the verifier refused a coin\\n");
                abort();
            }
            live_buf.clear();
            for (size_t i = 0; i < n / 8; i++) le64(out + 8 * i, w[i]);
            return;
        }
""")
    old = "        if (bits != 0) { while (!has_leading_zero_bits(sd, nonce, bits)) nonce++; }"
    assert body.count(old) == 1
    body = body.replace(old, "        if (bits != 0 && !live_cb) { while (!has_leading_zero_bits(sd, nonce, bits)) nonce++; }")
    assert "hasher.update" not in body.replace("hasher.update(d, n);", ""), "an absorb path still bypasses upd()"
    s = s[:body_start] + body + s[body_end:]
    if "#include <cstdio>" not in s:
        s = "#include <cstdio>\n#include <cstdlib>\n" + s
    ch.write_text(s)
    print("patched challenger.hpp")

pg = pathlib.Path("cuda-ghash/pow_grind.cuh")
s = pg.read_text()
if "live_cb" not in s:
    old = "    if (bits == 0) {\n        nonce = 0;"
    assert s.count(old) == 1
    s = s.replace(old, "    if (bits == 0 || challenger.live_cb) {\n        nonce = 0;")
    pg.write_text(s)
    print("patched pow_grind.cuh")

pf = pathlib.Path("cuda-ghash/prove_ffi.cu")
s = pf.read_text()
if "live_cb" not in s:
    old = "    const char* dump_z_path;\n} FlockCudaProveParams;"
    assert s.count(old) == 1
    s = s.replace(old, "    const char* dump_z_path;\n    FsChallenger::LiveCb live_cb;\n    void* live_ctx;\n} FlockCudaProveParams;")
    old = "    FsChallenger ch(P->domain, P->domain_len);\n"
    assert s.count(old) == 1
    s = s.replace(old, old + "    if (P->live_cb) ch.enable_live(P->live_cb, P->live_ctx, P->domain, P->domain_len);\n")
    pf.write_text(s)
    print("patched prove_ffi.cu")

t = pathlib.Path("crates/flock-cuda-ffi/tests/gpu_roundtrip.rs")
s = t.read_text()
assert "G128RESULT" in s, "run flock-128's g128_patch.py first"
if "GPU_LIVE" not in s:
    old = "    dump_z_path: *const c_char,\n}"
    assert s.count(old) == 1
    s = s.replace(old, "    dump_z_path: *const c_char,\n    live_cb: Option<LiveCb>,\n    live_ctx: *mut std::ffi::c_void,\n}")
    old = "        dump_z_path: dump_c.as_ref().map_or(null(), |c| c.as_ptr()),\n"
    assert s.count(old) == 1
    s = s.replace(old, old + "        live_cb: live_cb_now(),\n        live_ctx: LIVE_CTX.load(std::sync::atomic::Ordering::SeqCst) as *mut std::ffi::c_void,\n")
    old = "static GPU_TEST_LOCK: Mutex<()> = Mutex::new(());\n"
    s = s.replace(old, old + '''
type LiveCb = unsafe extern "C" fn(*mut std::ffi::c_void, *const u8, usize, u32, *mut u64) -> i32;
static LIVE_CTX: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
unsafe extern "C" fn live_cb(ctx: *mut std::ffi::c_void, content: *const u8, len: usize, n: u32, out: *mut u64) -> i32 {
    let ch = unsafe { &mut *(ctx as *mut flock_live::LiveChallenger) };
    let c = unsafe { from_raw_parts(content, len) }.to_vec();
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ch.round(c, n as usize))) {
        Ok(coins) => {
            for (i, x) in coins.iter().enumerate() {
                unsafe {
                    *out.add(2 * i) = x.lo;
                    *out.add(2 * i + 1) = x.hi;
                }
            }
            0
        }
        Err(_) => -1,
    }
}
fn live_cb_now() -> Option<LiveCb> {
    (LIVE_CTX.load(std::sync::atomic::Ordering::SeqCst) != 0).then_some(live_cb as LiveCb)
}
fn live_on() -> bool {
    std::env::var("GPU_LIVE").is_ok_and(|s| s == "1")
}
fn live_table() -> String {
    std::env::var("VU_NAME").unwrap_or_else(|_| "blake3".into())
}
/// One live session: hello (the verifier checks it against its own configuration), the link context, then the reps.
fn live_begin() -> flock_live::SharedTransport {
    use flock_live::*;
    let addr = std::env::var("LIVE_VERIFIER").unwrap_or_else(|_| "127.0.0.1:7100".into());
    let t = shared(TcpTransport::connect(&addr).expect("connect to the live verifier"));
    let spec = TableSpec { name: live_table(), binding: Binding::R1cs { statement_digest: [0; 32] } };
    let hello = SessionConfig::flock_128_r2(vec![spec]).hello();
    assert!(matches!(t.lock().unwrap().call(Req::Hello(hello)), Ok(Resp::Ok)), "verifier refused the session parameters");
    let mut link = b"flock-live/link-stub/v1".to_vec();
    link.extend_from_slice(&7u64.to_le_bytes());
    assert!(matches!(t.lock().unwrap().call(Req::Link(link)), Ok(Resp::Ok)));
    t
}
fn live_prove_rep(t: &flock_live::SharedTransport, nbl: usize, rep: usize) -> GpuArtifacts {
    let mut ch = flock_live::LiveChallenger::new(t.clone(), &live_table(), rep, &g128_rep_domain(rep)).expect("open rep stream");
    LIVE_CTX.store(&mut ch as *mut _ as usize, std::sync::atomic::Ordering::SeqCst);
    let g = gpu_prove(nbl, None);
    LIVE_CTX.store(0, std::sync::atomic::Ordering::SeqCst);
    g
}
/// Send the reps' proofs (after `edit`) and return the verifier's verdict.
fn live_finish(t: &flock_live::SharedTransport, reps: &[GpuArtifacts], edit: &dyn Fn(usize, &mut R1csProofLigerito) -> Option<usize>) -> String {
    use flock_live::*;
    for rep in 0..reps.len() {
        let mut p = reps[rep].proof.clone();
        let src = edit(rep, &mut p).unwrap_or(rep);
        if src != rep {
            p = reps[src].proof.clone();
        }
        let bytes = bincode::serialize(&(&reps[src].commitment, &p)).unwrap();
        t.lock().unwrap().call(Req::Proof { table: live_table(), rep: rep as u32, bytes }).unwrap();
    }
    match t.lock().unwrap().call(Req::Finish) {
        Ok(Resp::Verdict(v)) => v,
        other => format!("{{\\"accepted\\":false,\\"why\\":\\"{other:?}\\"}}"),
    }
}
''')
    # timed runs: each run is one live session when GPU_LIVE=1
    old = """    for _ in 0..runs {
        last.clear();
        for rep in 0..reps {
            g128_set_rep(rep);
            last.push(gpu_prove(N_BLOCKS_LOG, None));
        }
        run_secs.push(last.iter().map(|g| g.prove_secs).sum());
    }
"""
    assert s.count(old) == 1
    s = s.replace(old, """    let mut verdicts: Vec<String> = Vec::new();
    for _ in 0..runs {
        last.clear();
        let sess = live_on().then(live_begin);
        for rep in 0..reps {
            g128_set_rep(rep);
            last.push(match &sess {
                Some(t) => live_prove_rep(t, N_BLOCKS_LOG, rep),
                None => gpu_prove(N_BLOCKS_LOG, None),
            });
        }
        run_secs.push(last.iter().map(|g| g.prove_secs).sum());
        if let Some(t) = &sess {
            let (sent, recv, calls) = t.lock().unwrap().stats();
            let wait_s = t.lock().unwrap().wait_s();
            let v = live_finish(t, &last, &|_, _| None);
            println!("G128LIVE {{\\"m\\":{},\\"netlist\\":\\"{}\\",\\"prove_s\\":{:.4},\\"wait_s\\":{wait_s:.4},\\"calls\\":{calls},\\"wire_sent\\":{sent},\\"wire_recv\\":{recv},\\"verdict\\":{v}}}",
                     last[0].r1cs.m, live_table(), run_secs.last().unwrap());
            verdicts.push(v);
        }
    }
    if live_on() {
        let m = last[0].r1cs.m;
        let ok = verdicts.iter().all(|v| v.contains("\\"accepted\\":true"));
        let mut sorted = run_secs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let bytes: Vec<usize> = last.iter().map(|g| bincode::serialize(&g.proof).unwrap().len() + bincode::serialize(&g.commitment).unwrap().len()).collect();
        println!(
            "G128RESULT {{\\"m\\":{m},\\"nbl\\":{},\\"profile\\":\\"{}\\",\\"reps\\":{reps},\\"live\\":true,\\"netlist\\":\\"{}\\",\\"warmup_s\\":{warmup_secs:.4},\\"prove_median_s\\":{:.4},\\"prove_min_s\\":{:.4},\\"prove_runs_s\\":{:?},\\"proof_bytes\\":{},\\"all_verified\\":{ok}}}",
            N_BLOCKS_LOG, g128_env("GPU_PROFILE", "fast"), live_table(), sorted[sorted.len() / 2], sorted[0], run_secs, bytes.iter().sum::<usize>()
        );
        assert!(ok || expect_reject, "a live session was rejected: {verdicts:?}");
        if expect_reject {
            println!("G128NEG {{\\"m\\":{m},\\"live\\":true,\\"negative\\":\\"witness_tampered\\",\\"rejected\\":{},\\"verdict\\":{}}}", !ok, verdicts[0]);
        }
        if g128_env("GPU_NEG", "0") == "1" && !expect_reject {
            type Edit = Box<dyn Fn(usize, &mut R1csProofLigerito) -> Option<usize>>;
            let cases: Vec<(&str, Edit)> = vec![
                ("rep0_proof_replayed_as_rep1", Box::new(|rep, _| (rep == 1).then_some(0))),
                ("yr0_bit_flipped_rep1", Box::new(|rep, p| { if rep == 1 { p.pcs_open.ligerito.final_proof.yr[0].lo ^= 1; } None })),
                ("zerocheck_round_flipped_rep0", Box::new(|rep, p| { if rep == 0 { p.zerocheck.multilinear_rounds[0].0.hi ^= 1; } None })),
            ];
            for (name, edit) in cases {
                let t = live_begin();
                let rs: Vec<GpuArtifacts> = (0..reps).map(|rep| { g128_set_rep(rep); live_prove_rep(&t, N_BLOCKS_LOG, rep) }).collect();
                let v = live_finish(&t, &rs, &*edit);
                assert!(v.contains("\\"accepted\\":false"), "live negative {name} ACCEPTED: {v}");
                println!("G128NEG {{\\"m\\":{m},\\"live\\":true,\\"negative\\":\\"{name}\\",\\"rejected\\":true,\\"verdict\\":{v}}}");
            }
        }
        return;
    }
""")
    t.write_text(s)
    print("patched gpu_roundtrip.rs for live sessions")

c = pathlib.Path("crates/flock-cuda-ffi/Cargo.toml")
s = c.read_text()
if "flock-live" not in s:
    s = s.replace("[dev-dependencies]\n", "[dev-dependencies]\nflock-live = { path = \"../flock-live\" }\n", 1)
    c.write_text(s)
    print("flock-cuda-ffi dev-depends on flock-live")
