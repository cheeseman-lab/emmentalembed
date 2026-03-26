#!/bin/bash
# Setup script for folding environments.
#
# Creates separate conda environments for Chai-1 and AF3 since they have
# conflicting dependencies. Run this on a GPU node for CUDA compilation.
#
# Usage:
#   bash scripts/setup_fold_env.sh chai    # Install Chai-1 env
#   bash scripts/setup_fold_env.sh af3     # Install AF3 env
#   bash scripts/setup_fold_env.sh all     # Install both

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

setup_chai() {
    echo "=== Setting up Chai-1 environment ==="
    eval "$(conda shell.bash hook)"

    conda create -n emmentalembed-chai -c conda-forge python=3.11 uv pip -y
    conda activate emmentalembed-chai

    # Install base package
    cd "${PROJECT_ROOT}"
    uv pip install -e ".[fold]"

    # Install Chai-1
    uv pip install chai_lab

    echo ""
    echo "=== Chai-1 environment ready ==="
    echo "Activate with: conda activate emmentalembed-chai"
}

setup_af3() {
    echo "=== Setting up AlphaFold3 environment ==="
    echo ""
    echo "NOTE: AF3 weights must be downloaded separately."
    echo "Request access: https://forms.gle/svvpY4u2jsHEwWYS6"
    echo ""

    eval "$(conda shell.bash hook)"

    conda create -n emmentalembed-af3 -c conda-forge python=3.11 uv pip -y
    conda activate emmentalembed-af3

    # Install base package
    cd "${PROJECT_ROOT}"
    uv pip install -e ".[fold]"

    # AF3 dependencies - TODO: finalize once installation method is confirmed
    echo "AF3 package installation is pending. Install manually once available."

    echo ""
    echo "=== AF3 environment ready (pending AF3 package) ==="
    echo "Activate with: conda activate emmentalembed-af3"
}

case "${1:-}" in
    chai)
        setup_chai
        ;;
    af3)
        setup_af3
        ;;
    all)
        setup_chai
        setup_af3
        ;;
    *)
        echo "Usage: bash scripts/setup_fold_env.sh {chai|af3|all}"
        exit 1
        ;;
esac
