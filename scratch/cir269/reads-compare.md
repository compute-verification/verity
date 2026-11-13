# Records' reads, main 8854b8249777 against #9660 1333bd53f693

Runs: r20261113-142346-aa58 (base, --records-only) and r20261113-142348-fa99 (head, step 1). Each dumped every
manifest row's record with its own tree's `cache.read` (tools/records-job.sh). 887 rows each (886 records and the roots
record) over 879 names: four TEMPLATE_PARAMS bindings of verity_flock.subcircuit_templates repeat (5, 2, 2, 3 rows), and
the dump kept one row a name, so 8 rows are not compared.

- No binding's record reads fewer entries at the head; no read at the base is missing at the head (any kind).
- New at the head: present verity.definitions.registry.catalog.circuit-check (345), present ...registry.catalog (344),
  units ...catalog.circuit-check#common (50), #* (1).
- Digest changed: circuit-check's own (targets.py#common 879, #_kernel_roots 50, #* 1; cache.py, cli.py, units.py 345);
  enums verity.definitions.registry 879 (its `catalog` member); modules registry.catalog 2, call_scope 1; registered * 2;
  three import-site resolutions: verity_vllm.circuit-check#common's site verity_vllm.query.program_view (336),
  verity.definitions.boolean's site registry.catalog (50), verity.definitions.word's site call_scope (1).
