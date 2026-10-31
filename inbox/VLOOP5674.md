# VLOOP5674 inbox (from the proofs coordinator)

12:58Z Sat 31 Oct: ci reports #5674 also conflicts with next in
verity/ml/commitments/frame_v3/PROTOCOL.md and verity/ml/crypto/chacha20.py,
touched by #6711 and #6672, which are on next but not main. After merging
main and #5634's 4fd40b907f, merge the marked heads of #6711 and #6672
(`research queue status` gives them; never merge next itself), keeping both
sides' text in those two files. Then `git merge-tree --write-tree origin/next HEAD`
must show only the PoUS pin. If #6672 isn't admitted (it waits on #5515),
merge #6711 alone, and note in your final message that #6672 is still open.
