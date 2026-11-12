import sys, importlib.util
spec = importlib.util.spec_from_file_location("rt", "tests/lint/test_repo_retired_terms.py")
m = importlib.util.module_from_spec(spec); sys.modules["rt"] = m; spec.loader.exec_module(m)
for p in sys.argv[1:]:
    s = m.sites(open(p).read(), p)
    print(p, len(s), s[:10])
