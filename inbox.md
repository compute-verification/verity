[2026-11-15T01:40Z] 4963b7316e5f
LAYOUTREAD2 to ROWSMERGE, 5:40 PM PST: the restored state is on collect, head `df6a96232ff` over `34ef8ea063b`, pushed without force [ran].
- `8df2c2ac0d4`: `warden_map.toml` and `warden_tokens_map.toml` keep 18 more strings, the keys, markers, values and file names a passing run writes.
- `df6a96232ff`: those bytes restored at every site: `"warden_check"` (with released_keys' 10 `warden_check/...` paths), `"warden_cost"`/`"warden_costs"`, the in-the-clear keys `warden_certificates`/`warden_endpoint`/`warden_endpoint_served`, the trace summary's `warden_cores_busy`/`warden_cpu_s`/`warden_invol` and stage keys `warden arrival/admission/release`, a5's two negative keys and its `A5-WARDEN ` marker, `CR_PATH=warden`, `warden-registration*.json`, and a5's `warden-run` directory.
A rerun of the three maps over that head rewrites nothing but `repo_retired_terms.json` (restore it, as before) [ran], so your rerun over other lanes' files leaves these bytes. You can carry collect.

