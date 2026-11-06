export const shq = (s: string) => `'${s.replaceAll("'", `'\\''`)}'`;

export type Shell = {
  run(o: { command: string }): PromiseLike<{ exitCode: number; stderr: string }>;
  writeTextFile(o: { path: string; content: string }): PromiseLike<unknown>;
};

export async function mustRun(session: Pick<Shell, "run">, command: string) {
  const r = await session.run({ command });
  if (r.exitCode !== 0) throw new Error(`${command.split(" ").slice(0, 3).join(" ")}: exit ${r.exitCode}: ${r.stderr.trim().slice(-300)}`);
}
