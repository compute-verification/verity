[2026-11-17T08:38Z] 49f754efac9d
ci 08:3xZ: stop the #10748 read at 829c00e8ced9; I ejected it. Its base 62a7 held #10754, which failed the gate, and the chain was rebuilt. top regenerates the move on d2fd9c00ba5d; read the new head's own commits over d2fd once it is marked. Meanwhile take #10673.

