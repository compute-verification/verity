#!/usr/bin/env bash
# nvcc for a ship build (ship.sh's frame): CUDA's nvcc with the same arguments, except that cicc, its device compiler,
# reads nothing that says where a file sits in the tree. cicc's PTX follows the lengths of the strings it is given (its
# decisions follow its heap's layout), and three of them follow a file's place:
#   - the line markers of the preprocessed text it compiles, which name every file as its #include spells it or as -I
#     finds it: here each names its file by base name alone (its line and its system-header flag stay, so cicc's
#     diagnostics keep their lines and stay silent on system headers);
#   - --orig_src_path_name, the source's absolute path, which cudafe++ also hashes into the module id that names every
#     internal-linkage symbol: here the base name under /k, a directory nothing opens;
#   - the directory and the environment cicc runs in: here TMPDIR, with PATH, LD_LIBRARY_PATH, LD_PRELOAD and TMPDIR
#     alone.
# So a source that moved, or whose header did, compiles to the cubin it compiled to before. The steps are nvcc's own,
# taken from its dry run and run in its order; everything but those three is as nvcc would have run it. Refuses a dry
# run it can't follow (no cicc step). Needs CUDA (the toolkit) and TMPDIR, which ship.sh's frame sets.
set -euo pipefail
: "${CUDA:?the toolkit, which ship_cuda sets}" "${TMPDIR:?which the frame of ship.sh sets}"
dry=$("$CUDA/bin/nvcc" "$@" -dryrun 2>&1) || { echo "$dry" >&2; exit 1; }
grep -q '^#\$ .*/cicc" ' <<< "$dry" || { echo "ship_nvcc.sh: nvcc's dry run has no cicc step: $*" >&2; exit 1; }
# nvcc's temporaries, which it would have removed itself
temps=$(grep -o 'tmpxft_[0-9a-f]*_[0-9a-f]*' <<< "$dry" | sort -u)
trap 'for t in $temps; do rm -f "$TMPDIR/$t"-* "./$t"-*; done' EXIT
while IFS= read -r line; do
  case $line in '#$ '*) line=${line#'#$ '} ;; *) continue ;; esac
  if [[ $line =~ ^([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]]; then   # nvcc's own variables, which its command lines name
    value=${BASH_REMATCH[2]}; value=${value%"${value##*[![:space:]]}"}
    case ${BASH_REMATCH[1]} in
      PATH|LD_LIBRARY_PATH) export "${BASH_REMATCH[1]}=${value//\"/}" ;;
      *) printf -v "${BASH_REMATCH[1]}" %s "$value" ;;
    esac
    continue
  fi
  line=$(sed -E 's|--orig_src_path_name "[^"]*/([^"/]*)"|--orig_src_path_name "/k/\1"|' <<< "$line")
  case $line in
    *'/cicc" '*)
      ii=$(grep -o '"[^"]*\.ii"' <<< "$line" | tr -d '"')
      sed -E -i 's|^# ([0-9]+) "[^"<]*/([^"/]*)"|# \1 "\2"|' "$ii"
      (cd "$TMPDIR" && env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LD_PRELOAD="${LD_PRELOAD:-}" \
        TMPDIR="$TMPDIR" "$BASH" -c "${line//\$CICC_PATH/$CICC_PATH}") ;;
    'rm '*) eval "rm -f ${line#rm }" ;;   # nvcc's own rm doesn't mind a file it never wrote
    *) eval "$line" ;;
  esac
done <<< "$dry"
