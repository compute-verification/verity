"""[E6] run a command, print `[rss_run <tag>] rc= wall_s= peak_rss_gb=` (children's ru_maxrss; Linux KB, macOS bytes), exit with its rc."""
import resource, subprocess, sys, time
tag, cmd = sys.argv[1], sys.argv[2:]
t0 = time.monotonic()
rc = subprocess.call(cmd)
ru = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
gb = ru / (1 << 30) if sys.platform == "darwin" else ru / (1 << 20)
print(f"[rss_run {tag}] rc={rc} wall_s={time.monotonic() - t0:.1f} peak_rss_gb={gb:.2f}", flush=True)
sys.exit(rc)
