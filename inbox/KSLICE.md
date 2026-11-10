# Inbox: KSLICE (HASH-104)

## Tue 10 Nov 15:31Z: which machines your read-only runs may touch before 16:45Z

top's staged check runs its shards on `vy-nebius-cpu-6` and `vy-nebius-cpu-11` until about 16:40Z, and its audit on
`vy-nebius-cpu-5` until 16:44Z; its reds go to owners at 16:40Z and their fixes are due before the cut, so those
cores come first (top and infra on TOP-394, 15:29Z and 15:30Z). Your greps of run directories are light, but:

- **Before 16:45Z, launch nothing on cpu-5, cpu-6 or cpu-11.** Search the other machines first (cpu-1, cpu-2, cpu-3,
  cpu-8, cpu-12, and `vy-spot-glm53`), and those three after 16:45Z.
- One run a machine, a single `grep` process with `nice -n 19`, no `find` over anything but the runs directory.

## Tue 10 Nov 17:50Z: #8666 and #8669 are marked; the layout does not land today, so the rest waits

- The lead marked #8666 (9eb81cffe9) and #8669 (b0ad346b9f) at 17:47Z after reading their suites' lines (…adf4:
  `repository` 282, `verity-graph` 206; …e0b7: `repository` 282). #8666's body now names the live goal
  (`kernels/code-at-overhaul-bar`). Push nothing to either.
- top, TOP-394 17:39Z: Daniel wants the target tree changed first, so the layout is not on `main` today and the map
  changes once more. Piece 3 and the archive pull request (34 paths) both need the layout's `archive/`, so they
  wait for the new map's date. No restack. Do not delete or rewrite the hand-row files or the two lists in the notes.
- Nothing is asked of you now. top posts an architecture document for each directory's owner to correct; if the
  lead resumes you, it is to read that draft against your slice list.
