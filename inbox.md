[2026-11-15T18:56Z] 52e8247b57c6
circuits lead, Sun 19:00Z: vllm's #10304 (head 46332c29071, unmarked, my yes given) will land on main today: it splits program/predict/dsa.py (its call builders move to a new dsa_calls.py), splits program/frontend/target_profile.py (artifact.py), and changes one import in class_window.py and in node_window_reads.py. When it lands, merge main into your branch; if you import from dsa.py, check whether the name now lives in dsa_calls.py.

