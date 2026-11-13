[2026-11-13T22:05Z] 9b8ce664b0d0
Coordinator: #9770 has LANDED on main (e4f8f845a124): verity/hardware/cards.py and nodes.py are gone; their readers are in verity.device (load -> load_card, number, rate, check_card, load_node, ...). #9579 imports `from verity.hardware.cards import load, number`: merge main, switch to verity.device, rerun its suites remotely, push (own identity), re-mark. Same check for #9509 and #9434 if they name verity.hardware.cards/nodes.

