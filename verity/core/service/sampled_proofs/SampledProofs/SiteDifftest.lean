import Lean.Data.Json
import Flock.Canon
import CoreProtocol.Site

/-!
# The Site's difftest: `Core.Site` behind `Difftest.serve`

`site-difftest` serves `table` to the core difftest harness, whose Python half (`tests/test_site_difftest.py`) holds it to
`verity.core.site`. The Site is core's (`CoreProtocol.Site`, in `core_protocol`); its port is here because this package
already requires the interface and the harness, and the registration log is the proof service's record. The functions,
on the JSON `verity.core.site` writes, with the deployed digests (`Registration.sha512`):

* `site_refusal`, `{"auditor", "log", "entry"}`: the entry held to the log, `{"refusal": name | null, "head", "digest"}`;
* `site_check`, `{"auditor", "log"}`: the log checked and folded, `{"site", "head"}`, or `{"refused": {"entry", "name"}}`;
* `site_views`, `{"log"}`: the views of the log's Site, unchecked: its `units` and `boundary`, and for each of its
  locations' ids, `outside` and `nowhere`, its `sheet`, `device`, `node`, `tiers`, `boundary`, `registered` and `host`.
-/

namespace SampledProofs.SiteDifftest

open Lean (Json toJson)
open Flock (V)
open Core Core.Site Core.Site.Registration

def D : Digests := Registration.sha512

private def optJson {α : Type} (f : α → Json) : Option α → Json
  | none => .null
  | some x => f x

def siteRefusal (j : Json) : V Json := do
  exactly j ["auditor", "entry", "log"] "the input"
  let A ← Auditor.ofJson (← Flock.J.get j "auditor" "the input")
  let (c, log) ← Registration.ofJson (← Flock.J.get j "log" "the input")
  let e ← Registration.Entry.ofJson (← Flock.J.get j "entry" "the input")
  let h := head D c log
  pure (Json.mkObj [("digest", .str (D.entry e)), ("head", .str h),
    ("refusal", optJson (fun r : Refusal => .str r.name) (e.refusal D A c (Site.ofLog log) h))])

def siteCheck (j : Json) : V Json := do
  exactly j ["auditor", "log"] "the input"
  let A ← Auditor.ofJson (← Flock.J.get j "auditor" "the input")
  let (c, log) ← Registration.ofJson (← Flock.J.get j "log" "the input")
  match Registration.check D A c log with
  | .ok S => pure (Json.mkObj [("head", .str (head D c log)), ("site", S.toJson)])
  | .error (i, r) => pure (Json.mkObj [("refused", Json.mkObj [("entry", toJson i), ("name", .str r.name)])])

def ids (ks : List Link) : Json := .arr (ks.map fun k => Json.str k.id).toArray

def views (S : Site) (i : String) : Json :=
  Json.mkObj [
    ("boundary", ids (S.boundary (some i))),
    ("device", optJson (fun k : Sheet => Json.mkObj [("bw", toJson k.bw), ("chip", toJson k.chip),
      ("mem", toJson k.mem), ("tc", toJson k.tc)]) (S.device D.sheet i)),
    ("host", optJson .str (S.host i)),
    ("node", optJson Site.toJson (S.node D.sheet i)),
    ("registered", .bool (S.registered i)),
    ("sheet", optJson Sheet.toJson (S.sheetOf D.sheet i)),
    ("tiers", Json.mkObj ((S.tiers D.sheet i).map fun (t, n) => (t.name, toJson n)))]

def siteViews (j : Json) : V Json := do
  exactly j ["log"] "the input"
  let (_, log) ← Registration.ofJson (← Flock.J.get j "log" "the input")
  let S := Site.ofLog log
  pure (Json.mkObj [("boundary", ids (S.boundary none)),
    ("ids", Json.mkObj ((S.locs.map (·.id) ++ [outside, "nowhere"]).map fun i => (i, views S i))),
    ("units", .arr (S.units.map Json.str).toArray)])

def table : List (String × (Json → V Json)) :=
  [("site_refusal", siteRefusal), ("site_check", siteCheck), ("site_views", siteViews)]

end SampledProofs.SiteDifftest
