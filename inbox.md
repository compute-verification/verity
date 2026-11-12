[2026-11-12T23:45Z] bc2045527360
From the lead (compute-research), Thu 12 Nov 3:45 PM PST. Two facts changed, and the map's pull request is due now. If you are reading this as a resume, you are on a new machine: your map, hand patch and body draft are in the notes repo under lanes/compute-research/tools/seats/ (mover2-hardware_map.toml, mover2-hardware-by-hand.patch, mover2-hardware-map-body.draft.md).

1. #9442 is on main (landed Thu 12 Nov 3:29 PM PST; main is 7e70df68bd18). Take every TAKE AGAIN figure of the map's body on main now. Push the map's branch from main (one added file, tools/move/hardware_map.toml; name it cursor/compute-research-hardware-map-XXXX). Put the final body, with no TAKE AGAIN left and the draft comment gone, in the notes as tools/seats/mover2-hardware-map-body.md and sync (research notes sync --path lanes/compute-research). Open and mark nothing: I open it with your title and body and mark it.

2. Tonight top moves the Site's file (TOP-398, top's comment of 3:27 PM PST): verity/hardware/site/__init__.py becomes verity/device.py, and verity/configuration/deploy/, wire/ and config/ and verity/settings.py move too. It is due on main by Fri 13 Nov 8 AM PST; if it misses, it waits for Fri 7 PM PST. A mover leaves tools/move/** alone, so nothing re-keys your [split] row "verity/hardware/site/__init__.py: verity/hardware/cards". Answer these in your report, each tagged with how you know it:
   a. Does any test or check on main run a map under tools/move/ against the tree, so that top's move, landing with our map on main, would fail a suite because that row's file is gone? If yes: which, and say that the map should wait for top's move instead.
   b. What does the tool do at staging when that row's file is gone, and what is the smallest change that fixes it (the row re-keyed to verity/device.py)? Does it have to land as its own pull request before stage.py runs?
   c. Which lines of your hand patch does top's move rewrite? I count three places: the inputs lines of adapters/vllm/pyproject.toml and verity/hardware/pyproject.toml, and AGENTS.md's map.
   Say the answer to (b) in the map's body too, in one line, so its reader knows the row changes on Friday.

3. The move itself is staged Friday morning on whatever main is then: with top's move if it landed by 8 AM PST, without it if not. It still needs the map, #9375 (in doubt tonight: its head fails two tests and lean is pushing a fix) and #9415 (live since 3:19 PM PST). Do not stage tonight. I resume you Friday with main's SHA.

Return: the branch and head, the body's path and the notes commit, the answers to a, b and c, and what you ran with exits. Commit and push before you return.

