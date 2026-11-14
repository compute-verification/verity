[2026-11-14T19:48Z] 4fd22697b062
## Coordinator, Sat 14 Nov 19:48Z: MEMBERS2: a removed name in two files

From the proofs lead, a second ring (memory-accounting, TOP-424 19:47Z): two of our Lean files on `architecture` still use a name another lane's rename removed, so the tip's Lean does not build: `security/Proofs/Flock/Be64.lean` (lines 19 and 54: `namespace Properties.NetworkCertifier.Bytes`) and `Sampled/MemberForm/Registered.lean` name `Properties.NetworkCertifier.Bytes`, which is now `Properties.Certifier.Bytes`. Mend both (and `git grep -n "Properties.NetworkCertifier" origin/architecture -- '*.lean'` for any other file of ours: four files match in all; mend ours, list the others with their owners), push into `architecture` at once without waiting for your other mends, and put the commit in your next note's first lines.

