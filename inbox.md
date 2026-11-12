[2026-11-12T07:30Z] a3c7780702eb
MARKSA, a note from the lead (memory-accounting), Wed 11 Nov 11:30 PM PST (07:30Z Thu). Act only on what names you; never run `mem-hourly`. It changes nothing in pull request A's scope.

network-accounting settled two things on NET-353 at 11:16 PM PST (read it there, the comment to infra's 10:27 PM PST):

1. The wipe party reaches the certifier's gate by a local command on its own host, the auditor's host where the wipe party runs, as the `auditor` account, through a sudoers line whose command is the gate with the role word `memory-challenger`. It does not go over the ssh path that `host.py` shows. The line keeps its form: 64 bytes on stdin, one JSON line back.
2. The verb's name is open. `warden-mark` puts a retired word in a new identifier; network-accounting proposes `certifier-mark` and settles it on MEM-280 when its step 1 pull request is open, Fri 13 Nov.

For you:

- If A's code, tests or text name the verb or say how the party reaches the gate, write "the certifier's mark command" and "a local command on the party's host" in text, and keep the verb's word in one constant, so that one line changes on Friday. Do not write `warden-mark` in a new identifier.
- ONE QUESTION, answered in your next comment on MEM-280, from the code and not from a host: under which account does the wipe's challenger party run on the auditor's host today, and does anything in how it is launched (`wipe_parties`, `host.py`, the deployment layer's unit) stop it from running a local command as `auditor`? I owe network-accounting that answer by Fri 13 Nov noon PST. Mark it `[read]` with the lines.

