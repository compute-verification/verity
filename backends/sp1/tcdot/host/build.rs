//! Build the guest the host embeds, locked, with the two machine-specific path prefixes remapped as in
//! `../../host/build.rs` (`<backends/sp1>` -> `/verity/backends/sp1`, `<cargo home>` -> `/cargo`), so the ELF, and
//! with it the verifying key, is a function of the source tree, the lock file, the fork and the toolchain.

use std::env;
use std::path::PathBuf;

use sp1_build::{build_program_with_args, BuildArgs};

fn path_remaps() -> Vec<String> {
    let mut flags = Vec::new();
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"));
    let sp1_dir = manifest
        .join("../..")
        .canonicalize()
        .expect("host/ lives at backends/sp1/tcdot/host");
    flags.push(format!("--remap-path-prefix={}=/verity/backends/sp1", sp1_dir.display()));
    let cargo_home = env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")));
    if let Some(cargo_home) = cargo_home {
        let cargo_home = cargo_home.canonicalize().unwrap_or(cargo_home);
        flags.push(format!("--remap-path-prefix={}=/cargo", cargo_home.display()));
    }
    flags
}

fn main() {
    let features = match env::var_os("CARGO_FEATURE_STREAM_OPERANDS") {
        Some(_) => vec!["stream-operands".to_string()],
        None => Vec::new(),
    };
    build_program_with_args(
        "../guest",
        BuildArgs { rustflags: path_remaps(), locked: true, features, ..Default::default() },
    );
}
