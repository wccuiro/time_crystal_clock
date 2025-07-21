#!/usr/bin/env bash
if [ -z "$1" ]; then
  echo "Usage: $0 path/to/traj_XXXXX.zst"
  exit 1
fi

ZST_FILE="$1"
TMP_BIN="$(mktemp /tmp/traj.XXXXXX.bin)"
OUT_TXT="traj.txt"

# 1) Decompress to a temp file
zstd -d --stdout "$ZST_FILE" > "$TMP_BIN"

# 2) Use hexdump to print:
#    - 1×u32  (idx)
#    - 1×u8   (jump_type)
#    - 5×f64  (time_jump, psi0_re, psi0_im, psi1_re, psi1_im)
hexdump -e '1/4 "%u " 1/1 "%u " 5/8 "%f " "\n"' "$TMP_BIN" > "$OUT_TXT"

# 3) Clean up
rm "$TMP_BIN"

echo "Wrote text dump to $OUT_TXT"
