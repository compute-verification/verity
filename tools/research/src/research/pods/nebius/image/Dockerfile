# The one base image every job on the Nebius nodes runs in; build.sh builds it on a node and imports it into k3s, tagged with this
# file's hash. CUDA for the nodes' driver (580: CUDA 13.0), the Lean toolchain every Lake package pins (tests/test_lean_packages.py),
# the uv check pins (tests/test_uv_pin.py), py-spy and the system packages. Python dependencies are not in it: a job installs them
# from uv.lock at run time. Jobs run as uid 1000 (sky/cluster_up.sh), the image's user `verity`.
FROM nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:7c7413a56200486f71f181cad9310f6fd31b6bb21816ade15fc9c1e1e927a5c1

ARG UV_VERSION=0.12.20
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.34.0
ARG ELAN_VERSION=v4.2.4
ARG PY_SPY_VERSION=0.4.2

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 ELAN_HOME=/opt/elan PATH=/opt/elan/bin:$PATH
RUN apt-get update -q \
 && apt-get install -y -q --no-install-recommends build-essential ca-certificates curl git python3 ripgrep \
 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh \
 && UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin uv tool install --python /usr/bin/python3 "py-spy==$PY_SPY_VERSION" \
 && rm -rf /root/.cache
# uid 1000 is the base's `ubuntu`; ELAN_HOME is the user's so a job can install another toolchain whatever its HOME
RUN userdel -r ubuntu && useradd -m -u 1000 -s /bin/bash verity && install -d -o verity -g verity /opt/elan
USER verity
RUN curl -LsSf "https://github.com/leanprover/elan/releases/download/$ELAN_VERSION/elan-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xzC /tmp \
 && /tmp/elan-init -y --no-modify-path --default-toolchain "$LEAN_TOOLCHAIN" \
 && elan toolchain install "$LEAN_TOOLCHAIN" \
 && rm /tmp/elan-init
WORKDIR /home/verity
