module

public import CoreProtocol.Site
meta import CoreProtocol.Site
meta import Lean.Data.Json

@[expose] public section

namespace Core

namespace Site

open Lean (Json)

namespace Scratch

private def host : Loc := { id := "node0", kind := .host, parent := some "site0", chip := some 8002732032 }
private def ns : Figure := if h : 0 ≤ mkRat 1 1000000000 then ⟨_, h⟩ else ⟨0, by decide⟩
private def us : Figure := if h : 0 ≤ mkRat 1 1000000 then ⟨_, h⟩ else ⟨0, by decide⟩
private def root : Loc := { id := "site0", kind := .site, timeUnit := some ns }
private def dram (bw : Option Nat) (bytes : Nat := 1 <<< 31) : Loc :=
  { id := "dram0", kind := .memory, parent := some "node0", tier := some .dram, bytes, bw }
private def gpu : Loc := { id := "gpu0", kind := .device, parent := some "node0", sheet := "s" }
private def hbm (bw : Option Nat) : Loc :=
  { id := "hbm0", kind := .memory, parent := some "gpu0", tier := some .hbm, bytes := 1 <<< 30, bw }
private def auditor : Registration.Auditor :=
  { sheet := fun _ => none, tolerance := ⟨0, by decide⟩, key := fun _ => none,
    memory := [("dram0", ⟨"node0", 6554, 1 <<< 31⟩)], hosts := [("node0", { chip := 8002732032 })] }
private def empty : Site := ⟨[], [], []⟩
private def entry (xs : List Loc) : Entry := { locs := [root, host, gpu] ++ xs }

-- a host's leaf states the auditor's rate and bytes; a device's leaf states none
#guard decide (Entry.Rates auditor empty (entry [dram (some 6554), hbm none]))
-- refused: a host's leaf with no rate, with another rate, with a rate of 0 the auditor does not state, with other
-- bytes; a device's leaf with a rate; a host's leaf the auditor has no figures for
#guard !decide (Entry.Rates auditor empty (entry [dram none]))
#guard !decide (Entry.Rates auditor empty (entry [dram (some 6553)]))
#guard !decide (Entry.Rates auditor empty (entry [dram (some 6554) 1]))
#guard !decide (Entry.Rates auditor empty (entry [hbm (some 3200)]))
#guard !decide (Entry.Rates { auditor with memory := [] } empty (entry [dram (some 6554)]))
#guard !decide (Entry.Rates { auditor with memory := [("dram0", ⟨"node0", 0, 1 <<< 31⟩)] } empty (entry [dram (some 0)]))
-- the JSON round trip, with and without a rate
#guard [dram (some 6554), hbm none].all fun x => (Loc.ofJson x.toJson).toOption == some x
#guard (dram (some 6554)).toJson.compress ==
  "{\"bw\":6554,\"bytes\":2147483648,\"chip\":null,\"drift\":null,\"id\":\"dram0\",\"key\":\"\",\"kind\":\"memory\",\"owner\":\"prover\"," ++
  "\"parent\":\"node0\",\"scheme\":\"\",\"shared\":[],\"sheet\":\"\",\"tier\":\"dram\",\"time_unit\":null}"
private def auditorJson : Json :=
  Json.mkObj [("drift", .arr #[Json.mkObj [("drift", "3/5000"), ("id", "cert0")]]),
    ("hosts", .arr #[Json.mkObj [("cards", .arr #[Json.mkObj [("count", (8 : Nat)), ("sku", "k")]]),
      ("chip", (8002732032 : Nat)), ("id", "node0"),
      ("shared", .arr #[Json.mkObj [("bw", (50 : Nat)), ("tier", "disk")]]),
      ("ways", .arr #[Json.mkObj [("beta", (400000000000 : Nat)), ("id", "node0/eth0")],
        Json.mkObj [("beta", Json.null), ("id", "node0/eth1")]])]]),
    ("keys", .arr #[Json.mkObj [("id", "cert0"), ("key", "k0"), ("scheme", "lms-sha512/n64-w1-h20/v0")]]),
    ("memory", .arr #[Json.mkObj [("bw", (6554 : Nat)), ("bytes", (2147483648 : Nat)), ("host", "node0"),
      ("id", "dram0")]]), ("sheets", .arr #[]), ("tolerance", "0")]
#guard ((Registration.Auditor.ofJson auditorJson).toOption.map fun A => (A.memory, A.hosts)) ==
  some ([("dram0", ⟨"node0", 6554, 2147483648⟩)],
    [("node0", ⟨8002732032, [("k", 8)], [(.disk, 50)], [("node0/eth0", some 400000000000), ("node0/eth1", none)]⟩)])
#guard ((Registration.Auditor.ofJson auditorJson).toOption.map fun A => (A.key "cert0", A.scheme "cert0", A.scheme "x")) ==
  some (some "k0", some "lms-sha512/n64-w1-h20/v0", none)
#guard ((Registration.Auditor.ofJson auditorJson).toOption.map fun A => A.drift.map fun d => (d.1, toString d.2.1)) ==
  some [("cert0", "3/5000")]
-- the read rate of a host's leaf alone; with a device's leaf whose sheet is not registered there is none
#guard (⟨[], (entry [dram (some 6554), hbm none]).locs, []⟩ : Site).readRate (fun _ => "s") ["dram0"] == some 6554
#guard (⟨[], (entry [dram (some 6554), hbm none]).locs, []⟩ : Site).readRate (fun _ => "s") ["dram0", "hbm0"] == none

-- a disk's leaf under a host is held as a DRAM leaf is: the check reads no tier
private def disk (bw : Option Nat) : Loc :=
  { id := "disk0", kind := .memory, parent := some "node0", tier := some .disk, bytes := 1 <<< 40, bw }
private def withDisk : Registration.Auditor :=
  { auditor with memory := ("disk0", ⟨"node0", 28, 1 <<< 40⟩) :: auditor.memory }
#guard decide (Entry.Rates withDisk empty (entry [dram (some 6554), disk (some 28)]))
#guard !decide (Entry.Rates withDisk empty (entry [dram (some 6554), disk none]))
-- a host, a device, a certifier or a link's end states no rate
#guard !decide (Entry.Rates auditor empty { locs := [root, { host with bw := some 1 }] })
-- the refusal an entry gets is `figures`' own
#guard decide (Entry.Rates auditor empty { locs := [root] })

-- the unit of time: a location of kind `site` states it, and it is not 0; a second one states the same; no other
-- location states one
#guard decide (Entry.Timed empty { locs := [root, host] })
#guard decide (Entry.Timed empty { locs := [root, { root with id := "site1" }] })
#guard !decide (Entry.Timed empty { locs := [{ root with timeUnit := none }] })
#guard !decide (Entry.Timed empty { locs := [{ root with timeUnit := some ⟨0, by decide⟩ }] })
#guard !decide (Entry.Timed empty { locs := [root, { host with timeUnit := some ns }] })
#guard !decide (Entry.Timed ⟨[], [root], []⟩ { locs := [{ root with id := "site1", timeUnit := some us }] })
#guard (⟨[], [root, host], []⟩ : Site).timeUnit == some ns ∧ empty.timeUnit == none
-- a rate a second over ticks, rounded up once: 1.31 Mbit/s over a millisecond of nanoseconds, and over one
#guard (⟨[], [root], []⟩ : Site).over 1310000 1000000 == some 1310
#guard (⟨[], [root], []⟩ : Site).over 1310000 1 == some 1 ∧ empty.over 1 1 == none
-- a boundary link no certifier splits: refused onsite, taken where it says why, and the Site then is not mediated
private def bare (why : Option String) : Link :=
  { id := "eth0", ends := ("node0", outside), β := some 1000000000, latency := 0, unmediated := why }
#guard !decide (Entry.Covered .onsite empty { locs := [root, host], links := [bare none] })
#guard decide (Entry.Covered .onsite empty { locs := [root, host], links := [bare (some "a stand-in")] })
#guard decide (Entry.Covered .remote empty { locs := [root, host], links := [bare none] })
#guard (⟨[], [root, host], [bare (some "a stand-in")]⟩ : Site).mediated == false
#guard (⟨[], [root, host], [bare none]⟩ : Site).mediated
-- a link that states no capacity: taken only where it says no certifier splits it, and never at a capacity of 0
private def unrated (why : Option String) : Link := { bare why with β := none }
#guard decide (Entry.Kinds empty { locs := [root, host], links := [bare none] })
#guard decide (Entry.Kinds empty { locs := [root, host], links := [unrated (some "a stand-in")] })
#guard !decide (Entry.Kinds empty { locs := [root, host], links := [unrated none] })
#guard !decide (Entry.Kinds empty { locs := [root, host], links := [{ bare none with β := some 0 }] })
#guard (⟨[], [root, host], [unrated (some "a stand-in")]⟩ : Site).mediated == false
-- a host's leaf under another host than the auditor's figure names is refused
#guard !decide (Entry.Rates { auditor with memory := [("dram0", ⟨"node1", 6554, 1 <<< 31⟩)] } empty
  (entry [dram (some 6554)]))
-- a host's on-chip bytes: a host states the auditor's figure for it; refused with none, with another, or for a host
-- the auditor has no figure for; no other location states one
#guard decide (Entry.Chips auditor { locs := [root, host, gpu] })
#guard !decide (Entry.Chips auditor { locs := [root, { host with chip := none }] })
#guard !decide (Entry.Chips auditor { locs := [root, { host with chip := some 1 }] })
#guard !decide (Entry.Chips { auditor with hosts := [] } { locs := [root, host] })
#guard !decide (Entry.Chips auditor { locs := [root, host, { gpu with chip := some 1 }] })
#guard (Loc.ofJson host.toJson).toOption == some host
-- the auditor's inventory, of a whole Site: every memory leaf it has figures for under a host is registered under it,
-- and every card it lists for the host; not with the disk left out, with the leaf under a device, or with a card short
private def D : Registration.Digests := ⟨fun k => k.sku, fun _ => "", fun _ => ""⟩
private def card : Sheet := { figures := ⟨"k", ⟨0, by decide⟩, fun _ => ⟨0, by decide⟩⟩, bw := 1, chip := 1, mem := 1 <<< 30 }
private def gpus (n : Nat) : List Loc :=
  (List.range n).map fun i => { id := s!"gpu{i}", kind := .device, parent := some "node0", sheet := "k" }
private def node (xs : List Loc) : Site := empty.add { sheets := [card], locs := [root, host] ++ xs }
private def listed (n : Nat) : Registration.Auditor :=
  { withDisk with hosts := [("node0", { chip := 8002732032, cards := [("k", n)] })] }
#guard decide (Registration.Complete D withDisk (node [dram (some 6554), disk (some 28)]))
#guard !decide (Registration.Complete D withDisk (node [dram (some 6554)]))
#guard !decide (Registration.Complete D withDisk
  (node (gpus 1 ++ [dram (some 6554), { disk (some 28) with parent := some "gpu0" }])))
#guard decide (Registration.Complete D (listed 2) (node (gpus 2 ++ [dram (some 6554), disk (some 28)])))
#guard decide (Registration.Complete D (listed 2) (node (gpus 3 ++ [dram (some 6554), disk (some 28)])))
#guard !decide (Registration.Complete D (listed 2) (node (gpus 1 ++ [dram (some 6554), disk (some 28)])))
-- a host the auditor has no figures for has nothing to hold: its entry's own checks refuse it (`Chips`)
#guard decide (Registration.Complete D { auditor with memory := [], hosts := [] } (node []))
-- a host's shared rate for a tier: three disks at 28 bits a second each read 84 together, or the host's shared rate
-- where that is less; a disk leaf's bytes can be returned from memory, so the host's DRAM tier adds its rate, once,
-- whether or not its leaf is among them; a host states the auditor's shared rates
private def disks (shared : List (Tier × Nat)) : Site :=
  ⟨[], [root, { host with shared }, dram (some 6554)] ++ (List.range 3).map fun i =>
    { id := s!"disk{i}", kind := .memory, parent := some "node0", tier := some .disk, bytes := 1 <<< 40, bw := some 28 },
   []⟩
#guard (disks []).readRate (fun _ => "") ["disk0", "disk1", "disk2"] == some (84 + 6554)
#guard (disks [(.disk, 50)]).readRate (fun _ => "") ["disk0", "disk1", "disk2"] == some (50 + 6554)
#guard (disks [(.disk, 50)]).readRate (fun _ => "") ["disk0", "disk1"] == some (50 + 6554)
#guard (disks [(.disk, 50)]).readRate (fun _ => "") ["disk0"] == some (28 + 6554)
#guard (disks [(.disk, 100)]).readRate (fun _ => "") ["disk0", "disk1", "disk2"] == some (84 + 6554)
#guard (disks [(.disk, 50)]).readRate (fun _ => "") ["disk0", "disk1", "disk2", "dram0"] == some (50 + 6554)
#guard (disks [(.disk, 50)]).readRate (fun _ => "") ["dram0"] == some 6554
#guard (disks [(.disk, 50), (.dram, 6000)]).readRate (fun _ => "") ["disk0"] == some (28 + 6000)
private def sharing : Registration.Auditor :=
  { auditor with hosts := [("node0", { chip := 8002732032, shared := [(.disk, 50)] })] }
#guard decide (Entry.Chips sharing { locs := [root, { host with shared := [(.disk, 50)] }] })
#guard !decide (Entry.Chips sharing { locs := [root, host] })
#guard !decide (Entry.Chips auditor { locs := [root, { host with shared := [(.disk, 50)] }] })
#guard !decide (Entry.Chips { auditor with hosts := [("node0", { chip := 8002732032, shared := [(.disk, 0)] })] }
  { locs := [root, { host with shared := [(.disk, 0)] }] })
-- a clock's drift: one of the auditor's devices states the auditor's figure for its clock, or none where the auditor
-- has none; refused with another, at 1 or more, and on a location that is not the auditor's device
private def root2 : String := "hss-lms-sha512/n64-w1-h20/n64-w1-h20/v0"
private def tenth : Figure := if h : 0 ≤ mkRat 1 10 then ⟨_, h⟩ else ⟨0, by decide⟩
private def one : Figure := ⟨1, by decide⟩
private def cert (drift : Option Figure) : Loc :=
  { id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key := "k0", scheme := root2, drift }
private def keyed (drift : List (String × Figure)) (scheme : String := root2) : Registration.Auditor :=
  { auditor with key := fun i => if i == "cert0" then some "k0" else none,
                 scheme := fun i => if i == "cert0" then some scheme else none, drift }
-- a key's scheme: the auditor's for the device, and one the check knows; none on a location that is not its device
#guard root2 ∈ SCHEMES
#guard !decide (Entry.Owners (keyed [] "lms-sha512/n64-w1-h20/v0") { locs := [root, cert none] })
#guard !decide (Entry.Owners (keyed [] "x") { locs := [root, { cert none with scheme := "x" }] })
#guard !decide (Entry.Owners (keyed []) { locs := [root, { host with scheme := root2 }] })
#guard decide (Entry.Owners (keyed [("cert0", tenth)]) { locs := [root, cert (some tenth)] })
#guard decide (Entry.Owners (keyed []) { locs := [root, cert none] })
#guard !decide (Entry.Owners (keyed [("cert0", tenth)]) { locs := [root, cert none] })
#guard !decide (Entry.Owners (keyed []) { locs := [root, cert (some tenth)] })
#guard !decide (Entry.Owners (keyed [("cert0", one)]) { locs := [root, cert (some one)] })
#guard !decide (Entry.Owners (keyed []) { locs := [root, { host with drift := some tenth }] })
#guard (Loc.ofJson (cert (some tenth)).toJson).toOption == some (cert (some tenth))
#guard (Loc.ofJson ({ host with shared := [(.disk, 50)] } : Loc).toJson).toOption == some { host with shared := [(.disk, 50)] }
-- the JSON of a location that states the unit of time and of a link that says why it is not split
#guard [root, host].all fun x => (Loc.ofJson x.toJson).toOption == some x
#guard root.toJson.compress ==
  "{\"bw\":null,\"bytes\":0,\"chip\":null,\"drift\":null,\"id\":\"site0\",\"key\":\"\",\"kind\":\"site\",\"owner\":\"prover\"," ++
  "\"parent\":null,\"scheme\":\"\",\"shared\":[],\"sheet\":\"\",\"tier\":null,\"time_unit\":\"1/1000000000\"}"
#guard [bare none, bare (some "a stand-in"), unrated (some "a stand-in")].all fun k =>
  (Link.ofJson k.toJson).toOption == some k
#guard (unrated (some "a stand-in")).toJson.compress ==
  "{\"beta\":null,\"ends\":[\"node0\",\"outside\"],\"id\":\"eth0\",\"latency\":0,\"unmediated\":\"a stand-in\"}"
#guard (bare (some "a stand-in")).toJson.compress ==
  "{\"beta\":1000000000,\"ends\":[\"node0\",\"outside\"],\"id\":\"eth0\",\"latency\":0," ++
  "\"unmediated\":\"a stand-in\"}"
-- an auditor's host: at the site, the auditor's own, with its key; a certifier or a challenger may be in it; it is
-- under no host of the prover's, holds no device and no memory leaf, and no link ends at it
private def vm (parent : String := "site0") : Loc :=
  { id := "vm0", kind := .auditorHost, parent := some parent, owner := .auditor, key := "k1", scheme := root2 }
private def inVm : Loc := { cert none with parent := some "vm0" }
private def uplinks : List Link :=
  [{ id := "up0", ends := ("node0", "cert0"), β := some 1, latency := 0 },
   { id := "up1", ends := ("cert0", outside), β := some 1, latency := 0 }]
#guard decide (Entry.Kinds empty { locs := [root, host, vm, inVm], links := uplinks })
#guard decide (Entry.Kinds empty { locs := [root, host, vm, { inVm with kind := .challenger }] })
#guard !decide (Entry.Kinds empty { locs := [root, host, vm "node0"] })
#guard !decide (Entry.Kinds empty { locs := [root, host, { inVm with parent := some "node0" }], links := uplinks })
#guard !decide (Entry.Kinds empty { locs := [root, vm, { gpu with parent := some "vm0" }] })
#guard !decide (Entry.Kinds empty { locs := [root, vm, { dram none with parent := some "vm0" }] })
private def toVm : Link := { id := "l0", ends := ("node0", "vm0"), β := some 1, latency := 0 }
#guard !decide (Entry.Kinds empty { locs := [root, host, vm], links := [toVm] })
private def vmKeyed : Registration.Auditor :=
  { keyed [] with key := fun i => if i == "vm0" then some "k1" else (keyed []).key i,
                  scheme := fun i => if i == "vm0" then some root2 else (keyed []).scheme i }
#guard decide (Entry.Owners vmKeyed { locs := [root, vm, inVm] })
#guard !decide (Entry.Owners vmKeyed { locs := [root, { vm with owner := .prover }] })
#guard !decide (Entry.Owners vmKeyed { locs := [root, { vm with key := "k9" }] })
#guard !decide (Entry.Owners (keyed []) { locs := [root, vm] })
-- it states no on-chip bytes and no unit of time, and the auditor's inventory holds nothing of it
#guard decide (Entry.Chips auditor { locs := [root, host, vm] })
#guard !decide (Entry.Chips auditor { locs := [root, host, { vm with chip := some 1 }] })
#guard decide (Registration.Complete D withDisk (node [dram (some 6554), disk (some 28), vm, inVm]))
-- its JSON, and the views: it is no host of the prover's, a device in it is in none, and a challenger in it reaches
-- the hosts its links end at
#guard (Loc.ofJson vm.toJson).toOption == some vm
#guard (vm.toJson.getObjValAs? String "kind").toOption == some "auditor-host"
#guard (⟨[], [root, host, gpu, vm, inVm], uplinks⟩ : Site).host "cert0" == none
#guard (⟨[], [root, host, gpu, vm, inVm], uplinks⟩ : Site).host "vm0" == none
#guard (⟨[], [root, host, gpu, vm, inVm], uplinks⟩ : Site).host "gpu0" == some "node0"
#guard (⟨[], [root, host, vm, { inVm with id := "chal0", kind := .challenger }],
  [{ id := "mc0", ends := ("chal0", "node0"), β := some 1, latency := 0 }]⟩ : Site).reached.contains "node0"

end Scratch

end Site

end Core
