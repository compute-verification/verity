//! Build the guest the host embeds, so that the same source yields the same ELF -- and so the same
//! verifying key -- on every machine.
//!
//! `sp1_build` without Docker compiles the guest with the local `succinct` toolchain, and the ELF then
//! carries every absolute source path the compiler saw (panic locations in this workspace's crates and
//! in the registry sources of dependencies).  A checkout under `/Users/...`, `/workspace/research/src/<sha>/...`
//! or `/root/.cargo/registry/...` would each be a different guest, i.e. a different prover identity, for the
//! same code.  So the two host-specific prefixes are remapped (`--remap-path-prefix`) to fixed names:
//!
//!     <this workspace>/backends/sp1  ->  /verity/backends/sp1
//!     <cargo home>                   ->  /cargo
//!
//! and the build is `--locked` against the tracked `Cargo.lock`.  What remains in the ELF is a function of
//! the source tree, the lock file and the pinned toolchain (`sp1up --version v6.4.0`).  Cross-host
//! reproducibility is checked empirically (`veritor-zk-host info` on two machines); Docker builds
//! (`BuildArgs { docker: true }`) are the alternative when a machine can run them.
//!
//! The host's `unsound-*` features are forwarded to the guest build, so `cargo build -p veritor-zk-host
//! --features unsound-skip-authentication` yields a host whose embedded guest is the matching DIAGNOSTIC build
//! (and whose `info` names the feature).

use std::env;
use std::path::PathBuf;

use sp1_build::{build_program_with_args, BuildArgs};

const GUEST_FEATURES: &[(&str, &str)] = &[
    (
        "CARGO_FEATURE_UNSOUND_SKIP_AUTHENTICATION",
        "unsound-skip-authentication",
    ),
    (
        "CARGO_FEATURE_UNSOUND_SKIP_PATH_FOLD",
        "unsound-skip-path-fold",
    ),
    (
        "CARGO_FEATURE_UNSOUND_RELATION_ONLY",
        "unsound-relation-only",
    ),
    ("CARGO_FEATURE_ATTRIBUTION", "attribution"),
    ("CARGO_FEATURE_RELATION_BARE", "relation-bare"),
];

/// `--remap-path-prefix` flags for the two prefixes that differ between machines.
fn path_remaps() -> Vec<String> {
    let mut flags = Vec::new();
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .expect("host/ lives directly under the sp1 workspace")
        .canonicalize()
        .expect("the sp1 workspace directory exists");
    flags.push(format!(
        "--remap-path-prefix={}=/verity/backends/sp1",
        workspace.display()
    ));
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .or_else(|| env::var_os("USERPROFILE").map(|home| PathBuf::from(home).join(".cargo")));
    if let Some(cargo_home) = cargo_home {
        let cargo_home = cargo_home.canonicalize().unwrap_or(cargo_home);
        flags.push(format!("--remap-path-prefix={}=/cargo", cargo_home.display()));
    }
    flags
}

fn main() {
    let features: Vec<String> = GUEST_FEATURES
        .iter()
        .filter(|(variable, _)| env::var_os(variable).is_some())
        .map(|(_, feature)| feature.to_string())
        .collect();
    println!(
        "cargo:rustc-env=VERITOR_GUEST_FEATURES={}",
        features.join(",")
    );
    build_program_with_args(
        "../guest",
        BuildArgs {
            features,
            rustflags: path_remaps(),
            locked: true,
            ..Default::default()
        },
    );
}
