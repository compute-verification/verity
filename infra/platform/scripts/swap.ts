// Moves leads onto the platform and back, over its HTTP API. Needs PLATFORM_URL
// and PLATFORM_API_TOKEN in the environment.
//
//   swap.ts register LANE --channel C --charter FILE --doc FILE [--model M] [--harness claude-code|codex]
//                         [--budget USD] [--cron MINUTES] [--kick TEXT]
//   swap.ts all LEADS.json [--kick TEXT]     register every lead in the file (an array of register's fields)
//   swap.ts pause LANE                      rollback: the platform stops taking events for LANE
//   swap.ts resume LANE
//   swap.ts roll [LANE]                     move LANE (default: every active lead) to a fresh run on the current deployment
//   swap.ts doc LANE                        print LANE's workstream doc (to hand back to Cursor)
//   swap.ts status [LANE]
import { readFileSync } from "node:fs";

const url = process.env.PLATFORM_URL?.replace(/\/$/, "");
const token = process.env.PLATFORM_API_TOKEN;
if (!url || !token) {
  console.error("set PLATFORM_URL and PLATFORM_API_TOKEN");
  process.exit(2);
}

async function api(method: "GET" | "POST", path: string, body?: unknown) {
  const r = await fetch(`${url}${path}`, {
    method,
    headers: { authorization: `Bearer ${token}`, "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const text = await r.text();
  if (!r.ok) throw new Error(`${method} ${path}: ${r.status} ${text.slice(0, 500)}`);
  return JSON.parse(text) as Record<string, unknown>;
}

function flags(args: string[]) {
  const out: Record<string, string> = {};
  for (let i = 0; i < args.length; i++) {
    const a = args[i]!;
    if (!a.startsWith("--")) continue;
    out[a.slice(2)] = args[i + 1] ?? "";
    i++;
  }
  return out;
}

type Entry = {
  lane: string;
  slackChannel?: string;
  charterFile?: string;
  docFile?: string;
  model?: string;
  harness?: string;
  monthlyBudgetUsd?: number;
  cronEveryMinutes?: number;
};

const read = (f?: string) => (f ? readFileSync(f, "utf8") : undefined);

async function register(e: Entry, kick?: string) {
  const r = await api("POST", "/api/leads", {
    lane: e.lane,
    slackChannel: e.slackChannel,
    charter: read(e.charterFile),
    doc: read(e.docFile),
    model: e.model,
    harness: e.harness,
    monthlyBudgetUsd: e.monthlyBudgetUsd,
    cronEveryMinutes: e.cronEveryMinutes,
    status: "active",
    roll: true,
    kick,
  });
  console.log(JSON.stringify(r));
}

async function activeLanes(): Promise<string[]> {
  const s = (await api("GET", "/api/leads")) as { leads: { lane: string; status: string }[] };
  return s.leads.filter((l) => l.status === "active").map((l) => l.lane);
}

const [cmd, ...rest] = process.argv.slice(2);
const f = flags(rest);
const lane = rest.find((a, i) => !a.startsWith("--") && !rest[i - 1]?.startsWith("--"));

switch (cmd) {
  case "register":
    if (!lane) throw new Error("register LANE ...");
    await register(
      {
        lane,
        slackChannel: f.channel,
        charterFile: f.charter,
        docFile: f.doc,
        model: f.model,
        harness: f.harness,
        monthlyBudgetUsd: f.budget ? Number(f.budget) : undefined,
        cronEveryMinutes: f.cron ? Number(f.cron) : undefined,
      },
      f.kick,
    );
    break;
  case "all": {
    if (!lane) throw new Error("all LEADS.json");
    for (const e of JSON.parse(readFileSync(lane, "utf8")) as Entry[]) await register(e, f.kick);
    break;
  }
  case "pause":
  case "resume":
    if (!lane) throw new Error(`${cmd} LANE`);
    console.log(JSON.stringify(await api("POST", "/api/leads", { lane, status: cmd === "pause" ? "paused" : "active", kick: cmd === "resume" ? f.kick : undefined })));
    break;
  case "roll":
    for (const l of lane ? [lane] : await activeLanes()) console.log(JSON.stringify(await api("POST", "/api/leads", { lane: l, roll: true })));
    break;
  case "doc": {
    if (!lane) throw new Error("doc LANE");
    process.stdout.write(String((await api("GET", `/api/leads?lane=${encodeURIComponent(lane)}&doc=1`)).doc));
    break;
  }
  case "status":
    console.log(JSON.stringify(await api("GET", `/api/leads${lane ? `?lane=${encodeURIComponent(lane)}` : ""}`), null, 2));
    break;
  default:
    console.error("usage: swap.ts register|all|pause|resume|roll|doc|status ...");
    process.exit(2);
}
