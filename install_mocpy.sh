#!/bin/bash
# Installation script for custom MOCPy dependency
# This script handles the special installation requirements for the modified MOCPy

set -e  # Exit on any error

echo "Installing custom MOCPy dependency..."

# Check if Rust is installed
if ! command -v rustc &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rust-lang.org | sh
    source ~/.cargo/env
else
    echo "Rust is already installed."
fi

# Check if maturin is installed
if ! python -c "import maturin" &> /dev/null; then
    echo "Installing maturin..."
    pip install maturin
else
    echo "Maturin is already installed."
fi

# Build and install custom MOCPy
echo "Building and installing custom MOCPy..."
cd mocpy_syndiff
maturin develop --release
cd ..

echo "Custom MOCPy installation completed successfully!"
echo "You can now use the enhanced point-in-polygon filtering functionality."
