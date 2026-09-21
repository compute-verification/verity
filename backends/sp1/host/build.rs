//! Build the guest the host embeds.  The host's `unsound-*` features are
//! forwarded to the guest build, so `cargo build -p veritor-zk-host --features
//! unsound-skip-authentication` yields a host whose embedded guest is the
//! matching DIAGNOSTIC build (and whose `info` names the feature).  Without
//! them this is exactly `sp1_build::build_program("../guest")`.

use sp1_build::{build_program, build_program_with_args, BuildArgs};

const GUEST_FEATURES: &[(&str, &str)] = &[
    (
        "CARGO_FEATURE_UNSOUND_SKIP_AUTHENTICATION",
        "unsound-skip-authentication",
    ),
    (
        "CARGO_FEATURE_UNSOUND_SKIP_PATH_FOLD",
        "unsound-skip-path-fold",
    ),
    ("CARGO_FEATURE_ATTRIBUTION", "attribution"),
];

fn main() {
    let features: Vec<String> = GUEST_FEATURES
        .iter()
        .filter(|(variable, _)| std::env::var_os(variable).is_some())
        .map(|(_, feature)| feature.to_string())
        .collect();
    println!(
        "cargo:rustc-env=VERITOR_GUEST_FEATURES={}",
        features.join(",")
    );
    if features.is_empty() {
        build_program("../guest");
    } else {
        build_program_with_args(
            "../guest",
            BuildArgs {
                features,
                ..Default::default()
            },
        );
    }
}
