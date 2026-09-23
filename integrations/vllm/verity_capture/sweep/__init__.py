"""verity_capture.sweep — R12 broad configuration sweep helpers.

Pure (torch-free).  `workload`: deterministic heterogeneous workload generator (lane bc).  `record`: CLI that writes /
updates one row or failure record (lane bc; used by `row_pod.sh` on the pods).  `validate`: loads + validates
`out/gen/sweep/{rows,failures,fixes}/*.json` against SCHEMA.md; `render`: generates MATRIX.md / STATUS.md /
FINAL-REPORT.md / VALIDATION.md from those records only (report owner).
"""
