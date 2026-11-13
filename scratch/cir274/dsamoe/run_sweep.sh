# the window at every prompt length 87..150 at OUTS outputs, patched ($PATCH=1) or not
I=$RESEARCH_RUN_DIR/inputs
git rev-parse HEAD; [ "$PATCH" = 1 ] && git apply $I/fix-chain-ranges.diff && git diff --stat
uv sync -q --all-packages --extra torch-cpu </dev/null || exit 9
.venv/bin/python $I/p3_fixcheck.py $OUT $OUTS 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150; echo sweep rc=$?
