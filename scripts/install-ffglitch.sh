#!/bin/bash
# Install ffglitch for datamosh effects
# Usage: ./install-ffglitch.sh [install_dir]

set -e

FFGLITCH_VERSION="0.10.2"
INSTALL_DIR="${1:-/usr/local/bin}"

# Detect architecture
ARCH=$(uname -m)
case "$ARCH" in
    x86_64)
        URL="https://ffglitch.org/pub/bin/linux64/ffglitch-${FFGLITCH_VERSION}-linux-x86_64.zip"
        ARCHIVE="ffglitch.zip"
        ;;
    aarch64)
        URL="https://ffglitch.org/pub/bin/linux-aarch64/ffglitch-${FFGLITCH_VERSION}-linux-aarch64.7z"
        ARCHIVE="ffglitch.7z"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

echo "Installing ffglitch ${FFGLITCH_VERSION} for ${ARCH}..."

# Create temp directory
TMPDIR=$(mktemp -d)
cd "$TMPDIR"

# Download
echo "Downloading from ${URL}..."
curl -L -o "$ARCHIVE" "$URL"

# Extract
echo "Extracting..."
if [[ "$ARCHIVE" == *.zip ]]; then
    unzip -q "$ARCHIVE"
elif [[ "$ARCHIVE" == *.7z ]]; then
    # Requires p7zip
    if ! command -v 7z &> /dev/null; then
        echo "7z not found. Install with: apt install p7zip-full"
        exit 1
    fi
    7z x "$ARCHIVE" > /dev/null
fi

# Find and install binaries
echo "Installing to ${INSTALL_DIR}..."
find . -name "ffgac" -o -name "ffedit" | while read bin; do
    chmod +x "$bin"
    if [ -w "$INSTALL_DIR" ]; then
        cp "$bin" "$INSTALL_DIR/"
    else
        sudo cp "$bin" "$INSTALL_DIR/"
    fi
    echo "  Installed: $(basename $bin)"
done

# Cleanup
cd /
rm -rf "$TMPDIR"

# Verify
echo ""
echo "Verifying installation..."
if command -v ffgac &> /dev/null; then
    echo "ffgac: $(which ffgac)"
else
    echo "Warning: ffgac not in PATH. Add ${INSTALL_DIR} to PATH."
fi

if command -v ffedit &> /dev/null; then
    echo "ffedit: $(which ffedit)"
else
    echo "Warning: ffedit not in PATH. Add ${INSTALL_DIR} to PATH."
fi

echo ""
echo "Done! ffglitch installed."
