module

public import Proofs.Pouw.Lead.Commit.Credit
public import Proofs.Pouw.Lead.Commit.Hybrid
public import Proofs.Pouw.Lead.Commit.Rounds
public import Proofs.Pouw.Lead.Commit.RowBound

/-!
# The lead's commitments

The hashing credit the guarantee reads (`Credit`), the properties of the code's commitments and window rounds
(`RowBound`, `Rounds`), and the hiding hybrid (`Hybrid`).

Its proofs go in `Lead/Commit/`, its declarations in `Pouw.SecurityProofs.Lead.Commit`, and each file is imported
here.
-/

@[expose] public section
