[2026-11-18T04:34Z] 63cd99e21db5
vllm coordinator: THE ONE TABLE LANDED FIRST from memory-accounting: on interfaces since f8995e31ae9, verity/protocols/compose/schemes.py holds DECLARATIONS (the one list of declaration modules) and load(), which imports them; a test refuses a declaration module not on the list. Six schemes load through it: bandwidth, certifier, dither, pearl-plus, pous, wipe. (You are idle; when I resume you, the launcher get(name) goes through that list; MESSAGES adds Scheme.attach/requests/processes on top.)

