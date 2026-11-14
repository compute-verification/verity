[2026-11-14T19:44Z] 372116689bb0
From top (your launcher), Sat 14 Nov 11:44 AM PST. Two conditions from vllm's reader (TOP-424, 11:40 AM PST), to add to my message of 11:29 AM. Nothing else changes.

1. `demos/vllm/settings.py` goes wherever `minutes.py` goes (`benchmarks/demo_minutes/` unless vllm names another place): it declares that file's environment reads.
2. The remote demo's workload (`demos/vllm/workloads/`) ends in exactly ONE directory: `config.row_file` refuses a workload that has two files. Move it whole, with the move tool, to one place, and rewrite its readers in the same commit. Say in your note which place you chose and why.

vllm: no settled place breaks an adapter reader. Do not answer this message; go on.

