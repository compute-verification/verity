[2026-11-10T23:32Z] a5e339b8e128
From compute-research (your lead), Tue 10 Nov 3:35 PM PST. Two more questions of the same kind as your five, for the same note. Same deadlines. If they would make you late, list them as open instead.

Another worker found that the Lean host rows (hostH200, hostSm120 in Security/Definitions/Pouw/DitherInstance.lean) use figures no data sheet carries:
6. A SHA-512 rate a core for each host CPU. The machine model's host.sha512 rows have 2.9344e7 compressions a second a core (Xeon 8468) and 3.0116e7 (Xeon 6776P). Neither xeon-platinum-8468-2s.toml nor xeon-6776p-2s.toml has a key for it.
7. The cores a GPU that a node's VM gives: 8 on the H200 nodes, 12 on the RTX PRO 6000 node. Each sheet's `nodes` text says what the VM sees, but there is no key.
Also for your information, no action: the Lean rows' AMX rates differ from the sheets' product (rate.amx x clock.turbo x those cores) by +0.13% (H200 host) and -0.78% (RTX host).

For 6 and 7 answer as for the others: ADDED (key, unit, mark, source; a measurement must cite its art: id) or DECLINED with the reason. Edit no Lean.

