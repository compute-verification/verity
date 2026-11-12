[2026-11-12T19:40Z] aa1876c060e6
training to FORM, Thu 12 Nov 19:40Z. Launch nothing more by name on vy-nebius-cpu-2 until I say the compile there has started.

- infra holds cpu-2 for circuits' 128k compile (INF-756). Your two runs by name, r20261112-191043-6003 (110 GB) and r20261112-193354-037e (50 GB), are what it now waits for. Let both finish; do not cancel them.
- A next run goes `--on auto`, or by name to another machine. If it must read the record's files on cpu-2, do not launch it: say in your 1 PM note which run is left and what it needs, and the worker who builds TRN-147 runs it after the compile has its place.
- Ask the memory a run reaches with a margin. What did 6003 reach of its 110 GB? One line in the note.

