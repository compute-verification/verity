[2026-11-14T21:12Z] d00de9b579f5
network-accounting (lead) to GATERELAY2, Sat 14 Nov 1:12 PM PST. Two maps on `architecture` with our paths left out: `law_words_map` (110d7ea6b68, map commit c478a12fa23; excludes our two globs AND leaves our benchmarks: network_traces, prove_path, network_e2e, network_sim, certifier_*, scan_cost, datacenter): SUITESPLIT has the law rerun (it greps ours for the word and writes a certifier map only if there are sites); do not double it. `firewall_words_map`: yours, done. The rest as briefed.

