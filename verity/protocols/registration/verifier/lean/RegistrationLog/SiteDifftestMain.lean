module

public import Difftest
public import RegistrationLog.SiteDifftest

@[expose] public section

def main : IO Unit := Difftest.serve (Difftest.table RegistrationLog.SiteDifftest.table)
