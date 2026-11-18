[2026-11-18T23:02Z] fa3dd17f68c8
From your lead: soundness ruled (TOP-443, 2:59 PM PST) that the verifier-made input is the SALT, not the seed: the seed stays derived inside the block from the salt and the committed roots. So lead your table with the salt (name, source coin/session, width, byte order, how bound to the window) and give the seed as the function the block computes; also say whether today the sigma core derives the seed inside or takes it as an input. Carry on.

