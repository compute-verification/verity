//! The circuit pin at the command line: `verify` refuses a statement directory whose circuit files are not a pinned
//! circuit set of the claimed relation, before it parses them or reads the proof; `circuit-digest` reports the set.
//! (Which sets pass is pins.rs's unit tests, on the pinned digests themselves.)

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

// SHA-256 test vectors (FIPS 180-2): "abc", the empty string, 1000 x 'a'.
const ABC: &str = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
const EMPTY: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const A1000: &str = "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3";

/// A checker-v2 statement directory with 96 steps whose circuit files are not circuits at all.
fn statement(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("verity-gkr-verify-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("circuit.txt"), "abc").unwrap();
    std::fs::write(d.join("epilogue.txt"), "").unwrap();
    std::fs::write(d.join("chain.txt"), "a".repeat(1000)).unwrap();
    std::fs::write(d.join("manifest.json"), "{\"variant\": \"v2\", \"steps\": 96}\n").unwrap();
    d
}

fn run(args: &[&str]) -> Output { Command::new(env!("CARGO_BIN_EXE_verity-gkr-verify")).args(args).output().unwrap() }

fn text(o: &Output) -> String { format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr)) }

fn verify(d: &Path, extra: &[&str]) -> Output {
    let dir = d.to_str().unwrap();
    let proof = d.join("absent.bin");
    let json = d.join("verdict.json");
    run(&[&["verify", "--dir", dir, "--proof", proof.to_str().unwrap(), "--json", json.to_str().unwrap()], extra].concat())
}

#[test]
fn circuit_digest_reports_the_files_and_no_relation() {
    let d = statement("digest");
    let o = run(&["circuit-digest", "--dir", d.to_str().unwrap()]);
    let out = text(&o);
    assert_eq!(o.status.code(), Some(0), "{out}");
    for want in [format!("\"circuit.txt\": \"{ABC}\""), format!("\"epilogue.txt\": \"{EMPTY}\""), format!("\"chain.txt\": \"{A1000}\""),
                 "\"steps\": 96".into(), "\"variant\": \"v2\"".into(), "\"pinned_relation\": null".into()] {
        assert!(out.contains(&want), "{want} not in {out}");
    }
}

#[test]
fn verify_refuses_an_unpinned_statement_before_reading_the_proof() {
    let d = statement("unpinned");
    let o = verify(&d, &["--relation", "bf16-ampere"]);
    let out = text(&o);
    assert_eq!(o.status.code(), Some(1), "{out}");
    assert!(out.contains("\"accepted\": false") && out.contains("not a pinned bf16-ampere circuit set"), "{out}");
    assert!(out.contains(&format!("circuit.txt sha256 {}...", &ABC[..16])) && out.contains("\"relation\": \"bf16-ampere\""), "{out}");
    let json = std::fs::read_to_string(d.join("verdict.json")).unwrap();
    assert!(json.contains("\"accepted\": false") && json.contains("not a pinned bf16-ampere circuit set"), "{json}");
    // K is part of the circuit set: 48 steps of the same files is another relation, refused on steps as well
    std::fs::write(d.join("manifest.json"), "{\"variant\": \"v2\", \"steps\": 48}\n").unwrap();
    assert!(text(&verify(&d, &["--relation", "bf16-ampere"])).contains("steps 48, pinned 96"));
}

#[test]
fn verify_refuses_a_relation_this_build_does_not_pin() {
    let d = statement("unknown");
    let o = verify(&d, &["--relation", "no-such-relation"]);
    let out = text(&o);
    assert_eq!(o.status.code(), Some(1), "{out}");
    assert!(out.contains("no circuits are pinned for relation no-such-relation") && out.contains("bf16-ampere"), "{out}");
}

#[test]
fn verify_needs_the_claimed_relation_or_the_explicit_opt_out() {
    let d = statement("norelation");
    let o = verify(&d, &[]);
    assert_eq!(o.status.code(), Some(2), "{}", text(&o));
    assert!(text(&o).contains("--relation"), "{}", text(&o));
}
