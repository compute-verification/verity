import json,sys,os
for rid in sys.argv[1:]:
    f=f"/workspace/research/runs/{rid}/suites/verity_core_protocols_composition.json"
    if os.path.exists(f):
        d=json.load(open(f)); p=d.get("passed") or {}
        g=lambda k: p.get("tests/test_composition_difftest.py::"+k)
        print(json.dumps({"id":rid,"counts":d.get("counts"),"remote_agrees":g("test_remote_agrees"),"product_remotes":g("test_product_over_remotes_subprotocols_is_remote"),"onsite_agrees":g("test_onsite_agrees"),"product_onsites":g("test_product_over_onsites_subprotocols_is_onsite"),"sum":round(sum(p.values()))}))
