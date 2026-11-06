import { type ChildProcess, spawn as spawnChild } from "node:child_process";
import { mkdir, readFile, rm, stat as fsStat, readdir, writeFile, lstat } from "node:fs/promises";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { Readable, Writable } from "node:stream";
import type { Experimental_SandboxSession } from "@ai-sdk/provider-utils";

// A sandbox session over a local directory, for tests and local runs. It
// isolates no process: it exists so the real HarnessAgent path runs without
// Vercel. Commands see only PATH and a HOME inside the directory, so neither
// the host's environment nor its git config reaches them or is changed by them.
export function localSandboxSession(dir: string): Experimental_SandboxSession {
  // HOME is read after the cd into the directory, so it must not be relative.
  const root = resolve(dir);
  const abs = (p: string) => (isAbsolute(p) ? p : join(root, p));

  const readBinaryFile = async ({ path }: { path: string }) => {
    try {
      return new Uint8Array(await readFile(abs(path)));
    } catch (e) {
      if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
      throw e;
    }
  };

  const writeBinaryFile = async ({ path, content }: { path: string; content: Uint8Array }) => {
    await mkdir(dirname(abs(path)), { recursive: true });
    await writeFile(abs(path), content);
  };

  const spawn = (async (opts: { command: string; workingDirectory?: string; env?: Record<string, string>; stdin?: boolean; abortSignal?: AbortSignal }) => {
    const child: ChildProcess = spawnChild("bash", ["-lc", opts.command], {
      cwd: opts.workingDirectory ? abs(opts.workingDirectory) : root,
      env: { PATH: process.env.PATH ?? "/usr/bin:/bin", HOME: root, LANG: "C.UTF-8", ...opts.env } as unknown as NodeJS.ProcessEnv,
      stdio: [opts.stdin ? "pipe" : "ignore", "pipe", "pipe"],
      signal: opts.abortSignal,
    });
    const exited = new Promise<{ exitCode: number }>((resolve) =>
      child.on("close", (code) => resolve({ exitCode: code ?? 1 })),
    );
    child.on("error", () => {});
    return {
      pid: child.pid,
      stdout: Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>,
      stderr: Readable.toWeb(child.stderr!) as ReadableStream<Uint8Array>,
      ...(opts.stdin ? { stdin: Writable.toWeb(child.stdin!) as WritableStream<Uint8Array> } : {}),
      wait: () => exited,
      kill: async () => {
        child.kill("SIGTERM");
      },
    };
  }) as Experimental_SandboxSession["spawn"];

  return {
    description: `Local directory sandbox rooted at ${root}.`,
    readFile: async ({ path }) => {
      const bytes = await readBinaryFile({ path });
      if (!bytes) return null;
      return new ReadableStream<Uint8Array>({
        start(c) {
          c.enqueue(bytes);
          c.close();
        },
      });
    },
    readBinaryFile,
    readTextFile: async ({ path, startLine, endLine }) => {
      const bytes = await readBinaryFile({ path });
      if (!bytes) return null;
      const text = new TextDecoder().decode(bytes);
      if (startLine == null && endLine == null) return text;
      const lines = text.split("\n");
      return lines.slice((startLine ?? 1) - 1, endLine ?? lines.length).join("\n");
    },
    writeFile: async ({ path, content }) => {
      const chunks: Uint8Array[] = [];
      for await (const c of content as unknown as AsyncIterable<Uint8Array>) chunks.push(c);
      await writeBinaryFile({ path, content: Buffer.concat(chunks) });
    },
    writeBinaryFile,
    writeTextFile: async ({ path, content }) => writeBinaryFile({ path, content: new TextEncoder().encode(content) }),
    stat: async ({ path }) => {
      try {
        const s = await fsStat(abs(path));
        return { type: s.isFile() ? "file" : s.isDirectory() ? "directory" : "other", size: s.size };
      } catch {
        return null;
      }
    },
    listDirectory: async ({ path }) => {
      try {
        const entries = await readdir(abs(path), { withFileTypes: true });
        return entries.map((e) => ({
          name: e.name,
          type: e.isFile() ? "file" : e.isDirectory() ? "directory" : e.isSymbolicLink() ? "symlink" : "other",
        }));
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
        throw e;
      }
    },
    deleteFile: async ({ path }) => {
      try {
        const s = await lstat(abs(path));
        if (s.isDirectory()) throw new Error(`${path} is a directory`);
        await rm(abs(path));
        return true;
      } catch (e) {
        if ((e as NodeJS.ErrnoException).code === "ENOENT") return false;
        throw e;
      }
    },
    spawn,
    run: async (opts) => {
      const p = await spawn({ ...opts, stdin: false });
      const [stdout, stderr, { exitCode }] = await Promise.all([
        new Response(p.stdout).text(),
        new Response(p.stderr).text(),
        p.wait(),
      ]);
      return { exitCode, stdout, stderr };
    },
  };
}
