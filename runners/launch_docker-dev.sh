#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.
set -x # echo on

__usage="launch_docker-dev.sh - Launching yawen/flow:<tag>

Usage: launch_docker-dev.sh [options]

Note: You can specify FLOW_LOCAL_MNT environment variables
  to mount local repository.
"

TAG=""

while test $# -gt 0; do
  case "$1" in
  --python-req)
    TAG="python-req" # Pull the image from Docker Hub
    shift
    ;;
  -h | --help)
    echo "${__usage}"
    exit 0
    ;;
  *)
    echo "Unrecognized flag $1" >&2
    exit 1
    ;;
  esac
done

DOCKER_IMAGE="yawen/flow:${TAG}"
# Specify FLOW_LOCAL_MNT if you want to mount a local directory to the docker container
if [[ ${FLOW_LOCAL_MNT} == "" ]]; then
  FLOW_LOCAL_MNT="${HOME}/flow"
fi

# install imitation in developer mode
# CMD="pip install -e .[docs,parallel,test] gym[mujoco]" # copied from ci/build_and_activate_venv.sh
CMD="cd /flow && pip install -e ."

docker run -it --rm --init \
  -p 8899:8899 \
  -v "${FLOW_LOCAL_MNT}:/flow" \
  ${DOCKER_IMAGE} \
  /bin/bash -c "${CMD} && exec bash"
  # /bin/bash -c "exec bash && ${CMD}"
