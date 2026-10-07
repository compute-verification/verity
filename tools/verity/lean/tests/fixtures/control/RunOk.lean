/-! The controls' test (`runs` in `lean-audit.json`): it exits 0 when its data file says `ok`. -/

def main (args : List String) : IO UInt32 := do
  match args with
  | [path] => return if (← IO.FS.readFile path).trim == "ok" then 0 else 1
  | _ => return 2
