#!/bin/bash
# Local testing script for artdag
# Tests the 3-phase execution without Redis/IPFS

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ARTDAG_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="${ARTDAG_DIR}/test_cache"
RECIPE="${SCRIPT_DIR}/simple_sequence.yaml"

# Check for input video
if [ -z "$1" ]; then
    echo "Usage: $0 <video_file>"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/test_video.mp4"
    exit 1
fi

VIDEO_PATH="$1"
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    exit 1
fi

# Compute content hash of input
echo "=== Computing input hash ==="
VIDEO_HASH=$(python3 -c "
import hashlib
with open('$VIDEO_PATH', 'rb') as f:
    print(hashlib.sha3_256(f.read()).hexdigest())
")
echo "Input hash: ${VIDEO_HASH:0:16}..."

# Change to artdag directory
cd "$ARTDAG_DIR"

# Run the full pipeline
echo ""
echo "=== Running artdag run-recipe ==="
echo "Recipe: $RECIPE"
echo "Input: video:${VIDEO_HASH:0:16}...@$VIDEO_PATH"
echo "Cache: $CACHE_DIR"
echo ""

python3 -m artdag.cli run-recipe "$RECIPE" \
    -i "video:${VIDEO_HASH}@${VIDEO_PATH}" \
    --cache-dir "$CACHE_DIR"

echo ""
echo "=== Done ==="
echo "Cache directory: $CACHE_DIR"
echo "Use 'ls -la $CACHE_DIR' to see cached outputs"
