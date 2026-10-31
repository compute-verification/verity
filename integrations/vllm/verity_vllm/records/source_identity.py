"""The source-identity record (`research-infra/source-identity/v1`): the tree each package the integration runs resolved from,
which `pipeline.source_identity.document` writes and the runner's telemetry embeds whole (job.json `source_identity`).  No stage
reads it back, so it has no type here; this module owns its schema id."""

SCHEMA = "research-infra/source-identity/v1"
