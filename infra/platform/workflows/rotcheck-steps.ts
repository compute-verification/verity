import { runtime } from "../src/runtime.ts";

export type RotResult = { done: boolean; exitCode: number | null; tail: string };

const env = () => ({
  repo: process.env.PLATFORM_ROTCHECK_REPO,
  ref: process.env.PLATFORM_ROTCHECK_REF ?? "main",
  dir: process.env.PLATFORM_ROTCHECK_DIR ?? "platform",
  alertChannel: process.env.PLATFORM_ALERT_CHANNEL,
});

export async function startRotCheck(name: string): Promise<{ name: string; cmdId: string }> {
  "use step";
  const { Sandbox } = await import("@vercel/sandbox");
  const rt = await runtime();
  const e = env();
  if (!e.repo) throw new Error("PLATFORM_ROTCHECK_REPO is not set");
  const gh = await rt.credentials.github("platform-rotcheck");
  const sandbox = await Sandbox.getOrCreate({
    name,
    source: gh
      ? { type: "git", url: e.repo, revision: e.ref, depth: 1, username: "x-access-token", password: gh.token }
      : { type: "git", url: e.repo, revision: e.ref, depth: 1 },
    runtime: "node24",
    resources: { vcpus: 4 },
    timeout: 45 * 60_000,
    tags: { platform: "rotcheck" },
  });
  const cmd = await sandbox.runCommand({ cmd: "bash", args: [`${e.dir}/scripts/rot-check.sh`], detached: true });
  return { name, cmdId: cmd.cmdId };
}

export async function pollRotCheck(o: { name: string; cmdId: string }): Promise<RotResult> {
  "use step";
  const { Sandbox } = await import("@vercel/sandbox");
  const sandbox = await Sandbox.get({ name: o.name });
  const cmd = await sandbox.getCommand(o.cmdId);
  if (cmd.exitCode == null) return { done: false, exitCode: null, tail: "" };
  const out = await cmd.output("both");
  return { done: true, exitCode: cmd.exitCode, tail: out.split("\n").slice(-40).join("\n") };
}

export async function reportRotCheck(name: string, r: RotResult) {
  "use step";
  const { Sandbox } = await import("@vercel/sandbox");
  const rt = await runtime();
  const ok = r.exitCode === 0;
  const text = ok
    ? `platform rot check ${name}: passed against the newest SDK releases.`
    : `platform rot check ${name}: FAILED (exit ${r.exitCode ?? "none"}).\n\`\`\`\n${r.tail.slice(-3000)}\n\`\`\``;
  await rt.store.raiseAlert(`rotcheck:${name}`, text);
  const channel = env().alertChannel;
  if (channel) await rt.slack.post({ channel, text });
  const sandbox = await Sandbox.get({ name }).catch(() => null);
  await sandbox?.stop().catch(() => {});
  return { name, ok, exitCode: r.exitCode };
}
