import { createHmac, timingSafeEqual } from "node:crypto";

// Linear signs each webhook delivery with the webhook's secret (HMAC-SHA256 of
// the raw body, hex, in Linear-Signature) and stamps it (webhookTimestamp, in
// ms); anything unsigned, mis-signed or over a minute old is refused.
export function verifyLinearSignature(o: { secret: string; signature: string | null; body: string; nowMs: number }): boolean {
  if (!o.signature) return false;
  const a = Buffer.from(createHmac("sha256", o.secret).update(o.body).digest("hex"));
  const b = Buffer.from(o.signature);
  if (a.length !== b.length || !timingSafeEqual(a, b)) return false;
  const stamp = Number((JSON.parse(o.body) as { webhookTimestamp?: unknown }).webhookTimestamp);
  return Number.isFinite(stamp) && Math.abs(o.nowMs - stamp) <= 60_000;
}

export function signLinearBody(secret: string, body: string): string {
  return createHmac("sha256", secret).update(body).digest("hex");
}

export type LinearIssueEvent = {
  kind: "issue-created" | "issue-state";
  team: string;
  identifier: string;
  title: string;
  state: string | null;
  url: string | null;
};

// An issue's creation or change of state, what a lead follows on its team, as
// on Cursor; everything else (comments, edits, other types) is dropped.
export function parseLinearEvent(body: Record<string, unknown>): LinearIssueEvent | null {
  if (body.type !== "Issue") return null;
  const d = body.data as { identifier?: unknown; title?: unknown; team?: { key?: unknown }; state?: { name?: unknown } } | undefined;
  const from = body.updatedFrom as Record<string, unknown> | undefined;
  const kind = body.action === "create" ? "issue-created" : body.action === "update" && from && "stateId" in from ? "issue-state" : null;
  if (!kind || typeof d?.identifier !== "string") return null;
  return {
    kind,
    team: typeof d.team?.key === "string" ? d.team.key : d.identifier.split("-")[0]!,
    identifier: d.identifier,
    title: String(d.title ?? ""),
    state: typeof d.state?.name === "string" ? d.state.name : null,
    url: typeof body.url === "string" ? body.url : null,
  };
}
