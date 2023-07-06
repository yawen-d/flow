#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

__usage="build_image.sh - Building and pushing Docker image

Usage: build_image.sh [options] [tags]

options:
  -h, --help                show brief help
tags:
  base                      base stage image
  python-req                python-req stage image
"

KEYS=""
PUSH=0

while test $# -gt 0; do
  case "$1" in
  base)
    KEYS+="base "
    shift
    ;;
  python-req)
    KEYS+="python-req "
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

if [[ -z $KEYS ]]; then
  KEYS="latest"
  echo "No tag found in the arguments! Building default image yawen/flow:${KEYS}"
fi

for key in $KEYS; do
  echo "----- Building yawen/flow:${key} ..."
  BUILD_CMD="docker build --target ${key} -t yawen/flow:${key} ."

  # Build image
  ${BUILD_CMD}

done
