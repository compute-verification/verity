export interface Db {
  query<T = Record<string, unknown>>(text: string, params?: unknown[]): Promise<{ rows: T[] }>;
  close?(): Promise<void>;
}

// DATABASE_URL picks the backend: `pglite:` (in-process Postgres, for tests and
// local runs; `pglite:memory` or `pglite:<dir>`) or any postgres:// URL (Neon in
// production, or any Postgres the platform is moved to).
export async function openDb(url: string): Promise<Db> {
  if (url.startsWith("pglite:")) {
    const { PGlite } = await import("@electric-sql/pglite");
    const where = url.slice("pglite:".length);
    const pg = where === "" || where === "memory" ? new PGlite() : new PGlite(where);
    return {
      query: async <T>(text: string, params?: unknown[]) => {
        const r = await pg.query<T>(text, params as unknown[] | undefined);
        return { rows: r.rows };
      },
      close: () => pg.close(),
    };
  }
  const { default: pg } = await import("pg");
  const pool = new pg.Pool({ connectionString: url, max: 5 });
  return {
    query: async <T>(text: string, params?: unknown[]) => {
      const r = await pool.query(text, params);
      return { rows: r.rows as T[] };
    },
    close: () => pool.end(),
  };
}

// PGlite runs one statement list per exec; multi-statement DDL goes through here.
export async function execScript(db: Db, script: string): Promise<void> {
  const statements = script
    .split(/;\s*\n/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  for (const s of statements) await db.query(s);
}
