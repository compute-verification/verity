set -u
I=$RESEARCH_RUN_DIR/inputs
echo "REQBLOCK source $(git rev-parse HEAD)"
mkdir -p scratch/cir271 scratch/cir274
cp "$I/collect43.sh" "$I/collect43.py" scratch/cir271/ && cp "$I/collect_par.sh" "$I/collect_par.py" scratch/cir274/
sha256sum scratch/cir271/collect43.* scratch/cir274/collect_par.* | cut -c1-16,66-
bash "scratch/$DRIVER"
