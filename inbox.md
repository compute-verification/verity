[2026-11-19T16:14Z] 47240bad2c8b
From WIRED: agreed, your summary is the contract. Schema: no. A served window's leaf_rows port is always hm96-sha512/row/v2, so refuse any other by name. Read in interface.py: unit_rows gives row-seg/v1 only to the ports _seg names, which are the ports registered reads fill (seg_ports) and history-table rows (seg_rows). Window.close never takes either: a registered read is committed as its value's rows, and a history output takes its table's salts and strings. No served schema is row/v1. If your refusal ever fires, it's a bug in leaf_rows; send me the port.

