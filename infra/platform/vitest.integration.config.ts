import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { workflow } from "@workflow/vitest";
import { defineConfig } from "vitest/config";

// A fresh local world per invocation: the plugin clears only its own worker's
// files, and a hook left by an earlier invocation would outlive its run.
export default defineConfig({
  plugins: [workflow({ dataDir: mkdtempSync(join(tmpdir(), "platform-wf-")) })],
  test: {
    include: ["test/**/*.integration.test.ts"],
    testTimeout: 60_000,
    hookTimeout: 60_000,
  },
});
