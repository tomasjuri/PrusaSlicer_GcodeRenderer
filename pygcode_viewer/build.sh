#!/bin/bash
# Build script for pygcode_viewer

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parse arguments
BUILD_TYPE="Release"
INSTALL_DEV=0
RUN_TESTS=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --dev)
            INSTALL_DEV=1
            shift
            ;;
        --test)
            RUN_TESTS=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== Building pygcode_viewer ==="
echo "Build type: $BUILD_TYPE"

# Create build directory
mkdir -p build
cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
    -DPYGCODE_BUILD_PYTHON=ON

# Build
cmake --build . --parallel

echo ""
echo "=== Build complete ==="

# Install in development mode if requested
if [ $INSTALL_DEV -eq 1 ]; then
    echo ""
    echo "=== Installing in development mode ==="
    cd "$SCRIPT_DIR"
    pip install -e .
fi

# Run tests if requested
if [ $RUN_TESTS -eq 1 ]; then
    echo ""
    echo "=== Running tests ==="
    cd "$SCRIPT_DIR"
    python -m pytest tests/ -v
fi

echo ""
echo "Done!"
