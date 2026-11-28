#!/bin/bash
# Mesh and solve one case with OpenFOAM v2406 in Docker (opencfd/openfoam-run:2406).
# Usage: cfd/run_case.sh cfd/runs/<case> [more cases...]
# On a machine with OpenFOAM v2406+ installed natively, just run <case>/Allrun instead.
# (Ubuntu 24.04's packaged OpenFOAM v1912 can mesh but fails on any function object, so the
#  forces can't be recorded with it.)
set -e
IMAGE=${OPENFOAM_IMAGE:-opencfd/openfoam-run:2406}
for c in "$@"; do
  c=$(cd "$c" && pwd)
  echo "== $(basename "$c")  $(date -u +%H:%M:%S)"
  docker run --rm -v "$c":/case "$IMAGE" bash -c \
    'source /usr/lib/openfoam/openfoam2406/etc/bashrc; cd /case && ./Allrun'
done
