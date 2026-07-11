#!/usr/bin/env bash
set -euo pipefail

python -m train.calibrate_uncertainty "$@"
