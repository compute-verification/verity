[2026-11-10T02:48Z] c5b809f207df
From the coordinator, Mon 9 Nov 6:48 PM PST. Soundness answered PRF-491 already: yes to all four, with conditions. My comment on NET-285 after my rulings has them. For you:

1. The premises stand. Both ε terms are named in the period statement's error (NET-293's).
2. Statement 3's final form reads only `scanOf … = .ok` and the check's acceptance. "Every block sampled `all`" and `entryBits P s ≤ t.free` are each derived from acceptance or dropped. NET-284's check does not refuse a scan sampled in part; it charges it by the other rule and publishes which. So read the conclusion from the acceptance and the published term.
3. New condition on 26,400: "an egress unit reads only its view gates" must be a `scanOf` refusal or proved of what it accepts. `ScanViewCut`'s proof from `scanOf … = .ok` alone is that. If the proof needs anything `scanOf` does not refuse today, add it as a third `need` in job A's PR, and say which on NET-285 when the PR opens.
4. `argsApart` leaves every listed statement once `scanOf` refuses.

Lean (LEAN-240) has not answered yet.

