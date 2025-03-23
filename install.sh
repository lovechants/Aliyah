#!/usr/bin/env bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}▶ Installing Aliyah - ML Training Monitor${NC}"

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo -e "${RED}Cargo is not installed. Please install Rust first:${NC}"
    echo -e "${BLUE}curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null && ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Pip is not installed. Please install Python first.${NC}"
    exit 1
fi

# Determine pip command
PIP_CMD="pip"
if ! command -v pip &> /dev/null; then
    PIP_CMD="pip3"
fi

# Install Rust binary
echo -e "${GREEN}▶ Installing Rust binary...${NC}"
if [[ -d ".git" ]]; then
    # We're in a git repo, build locally
    cargo build --release
    echo -e "${GREEN}✓ Built Rust binary locally${NC}"
    echo -e "${GREEN}▶ Installing Python package${NC}"
    $PIP_CMD install -e python/
else
    # We're not in a git repo, install from published sources
    cargo install aliyah
    echo -e "${GREEN}✓ Installed Rust binary from crates.io${NC}"
    echo -e "${GREEN}▶ Installing Python package${NC}"
    $PIP_CMD install aliyah
fi

echo -e "${GREEN}✓ Installation complete!${NC}"
echo -e "${BLUE}▶ Usage: aliyah <script.py>${NC}"
