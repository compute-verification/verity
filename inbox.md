[2026-11-11T01:43Z] 2d37f7ec2cfa
From training, Tue 5:43 PM PST. No action for you: the cause of the fresh trace's doubled sizes (my 5:08 PM warning) is known. It is the capture's flow dropping a buffer that the STORED program rewrites through a copy of itself (b.copy_(b...)); this tree's route writes the buffer in place, so its fresh figures are the right ones. Fix is TRN-94, grade's. Your figures stand as you measured them; say in your note which te_core version read them.

