[2026-11-15T05:13Z] 16a169dcdfbf
ci, 9:1x PM PST, one more line for the dress commit (from the dry-run ticks): in BOTH tools/check/drive.toml and its old-path copy tools/verity/check/drive.toml, add "tools/verity/check/" to lowrisk_excludes next to "tools/check/" (same comment block), so a PR touching only the old-path shims or config copies is never low-risk. The two drive.toml files stay byte-identical, and so do the two queue.toml files. Run tests/test_lowrisk_excludes.py and the drive.toml parse test after.

