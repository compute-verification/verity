[2026-11-12T18:26Z] c3468b4bc444
From your lead (compute-research), Thu 12 Nov 10:26 AM PST. Two things for the plan you are writing, then a change after it.

1. For the plan. top asks on TOP-398 (10:18 AM PST), for the signed layout: before the data sheets move, the two loaders take the directory as an argument, defaulting to hardware/, and get a file name the mover will not delete (CI-533: a certified pure move can delete a non-empty __init__.py, the loss your trial met). I answered there: the per-file rows give the file names inside the move (your V2), and the directory argument is its own small pull request before the move, the default unchanged, on top of #9336 (which changes nodes/__init__.py and is in the queue). Put that pull request in the plan's order and sketch its lines (cards: skus, load; nodes: kinds, load; whatever else reads CARDS or NODES). The date I gave top: the move on main Fri 13 Nov 6 PM PST if #9375 is on main by Fri noon PST.

2. After the plan is posted, STOP: do not start the seven served-path moves. Daniel ruled today that a worker's model is named by its task (top's rule on TOP-402, 9:45 AM PST; Daniel confirmed it to me at 10:24 AM PST). Moving files is an Opus 5.5 worker's work, and a worker launched on another model finishes what it has in hand and is not resumed. So make your work a successor's before you end:
- push your scratch commits (hw-plan, hw-base) to branches of your own named cursor/compute-research-hw-move-trial-<4 hex> (trials, never opened as pull requests);
- commit under ~/.research/notes/lanes/compute-research/tools/ the map's rows for V2 as a file, and in your note every command that rebuilds V2 from main (the nine hand lines, move_check.py's command, the named tests);
- write in your note the state of the gate's move (#9353: the map is marked and in the queue; what MOVE READY needs) and of each of the seven served-path moves (what is staged, what waits on what), and the two runs' ids with their results or state;
- sync the notes, then end your turn with the last message I asked for.

