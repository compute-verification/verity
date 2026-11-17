# Locking the Nebius nodes down: design

How the store's commands remove a sealed run.

## Removing a sealed run

On a sealed run, `research retention gc`, `rm` and `research data evict --runs` pass the whole run dir to the root helper
once their own checks pass (store/sealed.py): gc and rm run `sudo -n vy-seal-runs delete DIR`, and evict runs
`vy-seal-runs evict DIR --min-bytes N`. The helper checks again, as root, that the dir sits directly under a runs root, is
sealed and still has custody. Evict removes only the files `.custody` lists, at the size it recorded and unchanged since.
A path inside a sealed run is refused.
