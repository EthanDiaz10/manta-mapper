#!/bin/bash
# Run one OpenFOAM command inside a case with the Docker image, e.g.
#   cfd/foam.sh cfd/runs/<case> foamToVTK -latestTime
set -e
IMAGE=${OPENFOAM_IMAGE:-opencfd/openfoam-run:2406}
c=$(cd "$1" && pwd); shift
docker run --rm -v "$c":/case "$IMAGE" bash -c \
  "source /usr/lib/openfoam/openfoam2406/etc/bashrc; cd /case && $*"
