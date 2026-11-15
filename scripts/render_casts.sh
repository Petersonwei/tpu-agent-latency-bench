#!/usr/bin/env bash
# Convert every media/casts/*.cast into a GIF (for the blog) and an MP4 (for the video edit).
# Needs: brew install agg ffmpeg. Idle time is already capped at record time; SPEED speeds up more.
set -euo pipefail
cd "$(dirname "$0")/.."
SPEED=${SPEED:-1}
mkdir -p media/gif media/mp4
for c in media/casts/*.cast; do
  n=$(basename "$c" .cast)
  agg --font-size 20 --speed "$SPEED" --theme monokai "$c" "media/gif/$n.gif" >/dev/null
  ffmpeg -y -loglevel error -i "media/gif/$n.gif" -movflags faststart -pix_fmt yuv420p \
    -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" "media/mp4/$n.mp4"
  echo "$n -> media/gif/$n.gif, media/mp4/$n.mp4"
done
