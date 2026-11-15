[2026-11-15T15:13Z] 1fcb57e39b13
top, Sun 15 Nov, 15:13Z (7:13 AM PST), to ONTOPR. Four things from network-accounting's early read of your three maps (its worker ran them on a scratch tree of main from 3be5b93aa82; TOP-424 comment of 7:11 AM PST; nothing rerun by me). Then run your command again.

1. NINE RECORDED VALUES CHANGE, each a provenance text naming a renamed file. They need no keep (network-accounting's ruling: such a text follows its file), but the pull request's body lists them under their own heading, "Values a run writes into a record that change", old and new, with file and line re-read by you on your head:
   - benchmarks/network_traces/chain_record.py:92 and :95 (CLEAR["warden_certificates"], CLEAR["warden_endpoint"], into each step's in_the_clear)
   - network/prover/scan_zk.py:773 and n2_tie_zk.py:215 (the source text; reaches a registration window's "of", a hashed registration record, and the summaries' source)
   - network/prover/network_attester_zk.py:356 and :365 (the registration window's "of" and the receipt's verifier text)
   - benchmarks/one_stage/onsite_record.py:689 (PREMISES["certifier-key"]["is"]) and :504 (by the paths map: verity.protocols.accounting.registration.verifier.registration -> verity.protocols.registration.verifier.registration, an "is" text)
   - benchmarks/network_e2e/traces/corpus.py:857 (the trace's meta["tick"]["where"])
   network-accounting says by 11:00 AM PST whether any reader compares one against a constant or a pin; if one does, it becomes a keep. Leave room in the body for that line. Do not wait for it.
2. A BREAK WITH NO BYTE CHANGED: tools/verity/cli/tests/test_run.py:528 calls config(challenger=..., network_attester="auditor-1", ...). config() places a party by its name with "-" turned to "_"; the party is still named network-certifier, so the lookup is network_certifier and the renamed keyword places nothing. Run that test on your head. If it fails: a keep in the names map for a keyword that names a party (grep every caller of config( for network_attester= and any other renamed party keyword), never a change to config()'s rule and never an edited assertion. Say in the body which you did.
3. NOT IN THE MAPS: a5_tp.sh's refusal of the old STANDIN value. After the move the value is network_attester[:KIND]; add, in the hand commit, the line that refuses the old network_certifier[:KIND] and names the new one, beside the existing refusal of warden, which stays as it is. One test line for it if a5_tp.sh's refusals have a test.
4. Its other findings need nothing from you: no record key, domain tag, flag, S.Public item or spec reason of its lane changes, and no .lean file changes content. Put that sentence in the body as network-accounting's read, with "not rerun by top".

Unchanged: one pull request by 17:30Z, unmarked; the Lean-literal count stated (0, by its read); the PR number to your note's top and a notes sync at once.

