FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu18.04 AS base
ARG DEBIAN_FRONTEND=noninteractive

# FROM continuumio/miniconda3:latest
# MAINTAINER Fangyu Wu (fangyuwu@berkeley.edu)

FROM base as python-req

# System
RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    curl \
    wget \
    git \
    ssh \
    parallel \
    python3.7 \
    python3-pip \
    libxml2 \
    software-properties-common \
    unzip \
    vim \
    tmux \
    virtualenv \
    rsync \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the PATH to the venv before we create the venv, so it's visible in base.
# This is since we may create the venv outside of Docker, e.g. in CI
# or by binding it in for local development.
ENV VIRTUAL_ENV=/venv
ENV PATH="/venv/bin:$PATH"
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

# python-req stage contains Python venv, but not code.
# It is useful for development purposes: you can mount
# code from outside the Docker container.

WORKDIR /flow
# Copy over just setup.py and dependencies (__init__.py and README.md)
# to avoid rebuilding venv when requirements have not changed.
COPY ./setup.py ./setup.py
COPY ./requirements.txt ./requirements.txt
COPY ./README.md ./README.md
COPY ./flow/__init__.py ./flow/__init__.py
COPY ./flow/version.py ./flow/version.py

# SUMO dependencies
RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
	swig \
	libgdal-dev \
	libxerces-c-dev \
	libproj-dev \
	libfox-1.6-dev \
	libxml2-dev \
	libxslt1-dev \
	openjdk-8-jdk \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install SUMO
COPY ./scripts/setup_sumo_ubuntu1804.sh ./scripts/setup_sumo_ubuntu1804.sh
RUN ./scripts/setup_sumo_ubuntu1804.sh
# ENV SUMO_HOME="$HOME/sumo"
# ENV PATH="$HOME/sumo/bin:$PATH"
# ENV PYTHONPATH="$HOME/sumo/tools:$PYTHONPATH"

# Create virtual environment
COPY ./runners/build_and_activate_venv.sh ./runners/build_and_activate_venv.sh
RUN ./runners/build_and_activate_venv.sh /venv \
    && rm -rf $HOME/.cache/pip
