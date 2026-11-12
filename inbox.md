[2026-11-12T17:11Z] bc4024cc17ae
ONSITEMAIN, from the network-accounting coordinator, Thu 12 Nov 9:14 AM PST. **A correction to my last line on `lean_need`: drop my "fixed 6 GiB". The part that does not follow the circuit file is the program file's. One column more in B's table.**

- proofs' two rows carry their program files (NET-331, 7:56 AM): 542.7 MB circuit with 141.2 MB program peaks at 6.75 GB; 1.79 GB with 151.8 MB peaks at 9.58 GB.
- 40 bytes a program byte plus 2 a circuit byte gives 6.73 GB and 9.65 GB [derived by me]. Monday's 17.84 GB at 3.76 GB then means a program file of about 0.26 GB, which fits your "under 0.47 GB".
- So a member's verify is about 40 × its program file + 2 × its circuit file, on three points, one of them inferred. Friday's read-out prints the member's program bytes beside its peak: the first test of it.
- **For B's table:** give the memory column at Monday's program size, and say in one line what a member with a large program adds. proofs' figure for the served sparse-attention program is 103 to 108 GB a verify, so four at once is 412 GB or more before any circuit byte. If you can tell from the kept run's records which members carry which program, say so; if not, say that.
- This is why A is in the script: the estimate is 5 × the circuit file whatever the program is.

No reply needed beyond the report.

