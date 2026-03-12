#!/bin/bash

# Compile script for TEE application

set -e

echo "Compiling MNIST Demo TEE Application..."
echo "========================================"

# Clean previous build
make clean 2>/dev/null || true

# Build
make

echo ""
echo "Build completed!"
echo ""
echo "To install the TA:"
echo "  sudo make install"
echo ""
echo "To test:"
echo "  sudo ./host/mnist_demo"
