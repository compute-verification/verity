//! Measurement only, never landed: Fast100 for m = 36 and 37 by upstream's own `derive_profile` (which
//! `gen_ligerito_configs` runs for m = 22..=35), each written beside the shipped TOMLs and reloaded; m = 35's derivation is
//! compared with its shipped TOML first, so the two new ones come from the derivation that made the shipped ones.
use std::{env::args, fs, path::Path};

use flock_prover::pcs::ligerito::{LigeritoProfile, LigeritoSecurityConfig};

fn main() {
    let dir = args().nth(1).expect("the configs directory");
    let dir = Path::new(&dir);
    let c35 = LigeritoSecurityConfig::derive_profile(35, LigeritoProfile::Fast100).expect("m=35").to_toml_string().expect("toml");
    let shipped = fs::read_to_string(dir.join("m35_fast100.toml")).expect("m35_fast100.toml");
    println!("m=35: derived {} the shipped TOML", if c35 == shipped { "equals" } else { "DIFFERS FROM" });
    for m in [36usize, 37] {
        let cfg = LigeritoSecurityConfig::derive_profile(m, LigeritoProfile::Fast100).unwrap_or_else(|e| panic!("m={m}: {e}"));
        let toml = cfg.to_toml_string().expect("serialize");
        LigeritoSecurityConfig::from_toml_str(&toml).unwrap_or_else(|e| panic!("m={m}: written toml fails reload: {e}"));
        fs::write(dir.join(format!("m{m}_fast100.toml")), &toml).expect("write toml");
        let queries: usize = cfg.levels.iter().map(|l| l.queries).sum();
        println!("m={m}: levels={} Σqueries={queries}", cfg.levels.len());
    }
}
