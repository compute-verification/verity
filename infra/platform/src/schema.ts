// The platform's record. Workflow run data is retained for days; this is what lasts.
// Idempotent: `migrate()` runs it on every cold start.
export const SCHEMA = `
create table if not exists leads (
  lane text primary key,
  charter text not null default '',
  doc text not null default '',
  doc_rev integer not null default 0,
  harness text not null default 'claude-code',
  model text not null,
  slack_channel text,
  cron_every_minutes integer,
  next_tick_at timestamptz,
  monthly_budget_usd numeric,
  gateway_key_id text,
  gateway_key_sealed text,
  status text not null default 'active',
  run_id text,
  run_started_at timestamptz,
  lease_turn text,
  lease_until timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);
alter table leads add column if not exists slack_inbox text;
alter table leads add column if not exists linear_team text unique;
alter table leads add column if not exists apis text not null default 'slack,linear';
create unique index if not exists leads_inbox on leads (slack_channel, slack_inbox) where slack_inbox is not null;

-- What else wakes a lead on Slack, as its subscriptions do on Cursor: a channel's
-- top-level posts (thread_ts '', set at registration, never lapsing), or every
-- reply in one thread (watch_thread, until expires_at).
create table if not exists watches (
  lane text not null references leads(lane),
  channel text not null,
  thread_ts text not null default '',
  expires_at timestamptz,
  primary key (lane, channel, thread_ts)
);
create index if not exists watches_where on watches (channel, thread_ts);

create table if not exists events (
  id bigserial primary key,
  lane text not null references leads(lane),
  source text not null,
  kind text not null,
  dedupe_key text unique,
  payload jsonb not null,
  reply_to jsonb,
  received_at timestamptz not null default now(),
  turn_id text
);
create index if not exists events_pending on events (lane, id) where turn_id is null;

create table if not exists turns (
  id text primary key,
  lane text not null references leads(lane),
  kind text not null default 'lead',
  worker_id text,
  run_id text,
  status text not null default 'running',
  harness text not null,
  model text not null,
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  input_tokens integer not null default 0,
  output_tokens integer not null default 0,
  cache_read_tokens integer not null default 0,
  cache_write_tokens integer not null default 0,
  reasoning_tokens integer not null default 0,
  cost_usd numeric not null default 0,
  final_text text,
  error text,
  doc_rev_before integer,
  doc_rev_after integer
);
create index if not exists turns_lane_time on turns (lane, started_at);

create table if not exists transcript_parts (
  turn_id text not null references turns(id),
  seq integer not null,
  type text not null,
  content jsonb not null,
  text text not null default '',
  tsv tsvector generated always as (to_tsvector('english', text)) stored,
  at timestamptz not null default now(),
  primary key (turn_id, seq)
);
create index if not exists transcript_parts_tsv on transcript_parts using gin (tsv);

create table if not exists doc_revisions (
  lane text not null references leads(lane),
  rev integer not null,
  doc text not null,
  turn_id text,
  at timestamptz not null default now(),
  primary key (lane, rev)
);

create table if not exists workers (
  id text primary key,
  lane text not null references leads(lane),
  task text not null,
  branch text,
  harness text not null,
  model text not null,
  status text not null default 'running',
  run_id text,
  reply_to jsonb,
  result text,
  started_at timestamptz not null default now(),
  ended_at timestamptz
);

create table if not exists jobs (
  id text primary key,
  lane text not null references leads(lane),
  turn_id text,
  spec jsonb not null,
  status text not null default 'submitted',
  callback_token text not null unique,
  external_id text,
  reply_to jsonb,
  result jsonb,
  submitted_at timestamptz not null default now(),
  finished_at timestamptz
);

create table if not exists alerts (
  key text primary key,
  text text not null,
  at timestamptz not null default now()
);

create table if not exists grants (
  id bigserial primary key,
  subject text not null,
  kind text not null,
  scope text not null default '',
  ref text not null default '',
  issued_at timestamptz not null default now(),
  expires_at timestamptz,
  revoked_at timestamptz
);
create index if not exists grants_subject on grants (subject, issued_at);
`;
