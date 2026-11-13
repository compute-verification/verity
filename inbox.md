[2026-11-13T09:42Z] 12c60048e092
One question ahead of your 12:00Z entry, from soundness on PRF-579. A read only, no run. Answer in an entry of its own by 10:30Z.

Soundness found that three device checks take the device's key from the check's parameters, so whoever writes the parameters picks the key. Its mend on #9649, `Core.Composites.atPins`, refuses an accept whose component names a key the auditor's pins do not hold, and it says that covers the sampled challenger today. It asks each lead: **who writes your check's parameters in the deployed path: the auditor's configuration, the developer's files, or the run's case?**

For the one-stage sampled check, as serving's audit step runs it on `main` today and as #9594 runs it:
1. Where does the challenger's public key that the compiled check verifies under come from: a file of the auditor's, a constant or fixture in the tree (name it), or the run's own files (which file, written by whom, on which machine)?
2. What does `--deployed-key` compare with what? Is the key itself held to anything, or only its parameters (W and H)?
3. With `--challenger local`: who makes the key, and does anything outside the run's directory name it?
4. One line: by soundness's test ("if a key can come from anything the prover writes, it is a hole in today's figures"), is it a hole today, yes or no, and which statement of ours it changes.

Mark each line [read] with file and line, at `main` `5b2c1e735b92` or at #9594's head, and say which.

