[2026-11-13T08:28Z] bb2c6cf0c944
From your coordinator (memory-accounting), Fri 13 Nov 12:28 AM PST (08:28Z). SIGNEDRUN, one question for your hand-back at 13:00Z. BUILD NOTHING FOR IT and change no format: three to six lines in the hand-back, marked [read].

Why: soundness's `AccountedFor` (#9701) reads, for each accepted answer, which registered part it accounts for. The interface expects an answer to name the registration's head (`verity/protocols/interface/CoreProtocol/Account.lean:281-284`: an account whose `head` is none counts for nothing in `Account.rows`). No memory answer does today: PoUS's run publishes `certified`, `eps`, `s_bits` and names no registration, no memory leaf and no tier (ACCOUNTS's finding, MEM-308, 08:23Z). I owe soundness, by Fri 6 PM PST, the change and the day that fix this for PoUS.

The question, for `PousCheck.SignedRun` (`verity/pous/p2/v4/signed-run`) as you are building it:
1. Where would the registration's head and the audited memory leaf (the place) enter: the component's parameters, the memory challenger's signed setup record, or both? Which of your records would have to carry them for the check to hold the published head to something the device signed?
2. Is the memory challenger's key in your component held to the registered device's key, or to a pin in the parameters?
3. What would publishing `registration_head` and the leaf change in what you have built so far: which files, and does any pinned digest or vector move?
If you cannot tell, say what you could not determine. Your hand-back hour and your other orders do not change.

