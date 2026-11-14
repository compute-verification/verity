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
# So after a source moved, or a header it includes did, cicc is handed the text, the arguments and the directory it was
# handed before, and gives the cubin it gave before. The program's own text is compiled as written, so it may not name
# a file: __FILE__ (and an assert, which expands to one) puts into the text the name of the file it stands in, as its
# #include spells it, and a move that changes that spelling would change the program. A text that holds, outside a system
# header, the name its line markers give the file it is in, or the source's, is refused before cicc runs, naming the file
# and the line. The steps are nvcc's own, taken from its dry run and run in its order; everything but those three is as
# nvcc would have run it. Refuses a dry run it can't follow: one with no cicc step, or whose cicc step names a variable
# other than CICC_PATH. Needs CUDA (the toolkit) and TMPDIR, which ship.sh's frame sets.
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
      # cicc's environment is emptied, so the one variable of nvcc's its step names is written in; a step that names
      # another is refused rather than run with it empty
      line=${line//\$CICC_PATH/$CICC_PATH}
      case $line in
        *[\$\`]*)
          echo "ship_nvcc.sh: cicc's step names a variable besides CICC_PATH ($(grep -o '[$`][{(]*[A-Za-z_0-9]*' <<< "$line" |
            sort -u | tr '\n' ' ')), and it runs with none of nvcc's: write that one into the step as CICC_PATH is" >&2
          exit 1 ;;
      esac
      ii=$(grep -o '"[^"]*\.ii"' <<< "$line" | tr -d '"')
      # a line marker is `# LINE "FILE" FLAGS`, 3 among the flags for a system header; __FILE__ is FILE, quoted
      named=$(awk '
        /^# [0-9]+ "/ { split($0, q, "\""); file = q[2]; n = $2 - 1; system_header = file ~ /^</ || q[3] ~ /(^| )3( |$)/
                        if (source == "" && file !~ /^</) source = file
                        next }
        { n++ }
        !system_header && source != "" && (index($0, "\"" file "\"") || index($0, "\"" source "\"")) {
          print file " line " n ": " substr($0, 1, 160); exit }
      ' "$ii")
      [ -z "$named" ] || {
        echo "ship_nvcc.sh: the text cicc compiles names a file, as __FILE__ or an assert leaves it, so the program" \
          "would follow where that file sits: take it out of $named" >&2
        exit 1; }
      sed -E -i 's|^# ([0-9]+) "[^"<]*/([^"/]*)"|# \1 "\2"|' "$ii"
      (cd "$TMPDIR" && env -i PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" LD_PRELOAD="${LD_PRELOAD:-}" \
        TMPDIR="$TMPDIR" "$BASH" -c "$line") ;;
    'rm '*) eval "rm -f ${line#rm }" ;;   # nvcc's own rm doesn't mind a file it never wrote
    *) eval "$line" ;;
  esac
done <<< "$dry"
