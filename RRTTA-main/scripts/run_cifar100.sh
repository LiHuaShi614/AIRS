#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
"${PYTHON:-python}" run_rrtta.py --dataset cifar100 "$@"
