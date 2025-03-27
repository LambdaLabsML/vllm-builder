ARG CUDA_VERSION=12.4.1
ARG IMAGE_DISTRO=ubuntu22.04
ARG PYTHON_VERSION=3.12

# ---------- Builder Base ----------
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS base

# Job scaling
ARG MAX_JOBS=32
ENV MAX_JOBS=${MAX_JOBS}
ARG NVCC_THREADS=2
ENV NVCC_THREADS=${NVCC_THREADS}

# Set arch lists for all targets
# 'a' suffix is not forward compatible but enables all optimizations
ARG TORCH_CUDA_ARCH_LIST="9.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG VLLM_FA_CMAKE_GPU_ARCHES="90a-real"
ENV VLLM_FA_CMAKE_GPU_ARCHES=${VLLM_FA_CMAKE_GPU_ARCHES}

# ARM64 linker optimization
ENV USE_PRIORITIZED_TEXT_FOR_LD=1

# Update apt packages and install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt upgrade -y
RUN apt install -y --no-install-recommends \
    curl \
    gcc-12 g++-12 \
    git \
    libibverbs-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    zlib1g-dev

# Set compiler paths
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

# Setup build workspace
WORKDIR /workspace

# Prep build venv
ARG PYTHON_VERSION
RUN uv venv -p ${PYTHON_VERSION} --seed --python-preference only-managed
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

FROM base AS build-base
RUN mkdir /wheels

# Install build deps that aren't in project requirements files
# Make sure to upgrade setuptools to avoid triton build bug
RUN uv pip install -U build cmake ninja pybind11 "setuptools<=76" wheel

# Route targets based on platform
FROM --platform=linux/amd64 build-base AS build-amd64
FROM --platform=linux/arm64 build-base AS build-arm64

# Handle arm64 torch build
FROM build-arm64 AS build-torch
RUN apt install -y --no-install-recommends nvpl0
RUN uv pip install -U patchelf scons

# Build ARM ComputeLibrary
ARG ACL_REF=v24.09
ENV ACL_ROOT_DIR=${PWD}/acl
ENV ACL_INCLUDE_DIR=${ACL_ROOT_DIR}/include
ENV ACL_LIBRARY=${ACL_ROOT_DIR}/build
RUN mkdir acl
RUN git clone https://github.com/ARM-software/ComputeLibrary.git
RUN cd ComputeLibrary && \
    git checkout ${ACL_REF} && \
    git submodule sync --recursive && \
    git submodule update --init --recursive -j 8
RUN cd ComputeLibrary && \
    scons Werror=1 -j 8 build_dir=${ACL_LIBRARY} debug=0 neon=1 opencl=0 os=linux openmp=1 cppthreads=0 arch=armv8a multi_isa=1 fixed_format_kernels=1 build=native

ENV LD_LIBRARY_PATH=${ACL_LIBRARY}:${LD_LIBRARY_PATH}
ENV BLAS=NVPL

ARG TORCH_REF=v2.6.0
ARG TORCH_BUILD_VERSION=2.6.0+cu124
ENV PYTORCH_BUILD_VERSION=${TORCH_BUILD_VERSION:-${TORCH_REF#v}}
ENV PYTORCH_BUILD_NUMBER=0
RUN git clone https://github.com/pytorch/pytorch.git
RUN cd pytorch && \
    git checkout ${PYTORCH_REF} && \
    git submodule sync --recursive && \
    git submodule update --init --recursive -j8 && \
    # Bump XNNPACK submodule ref to fix compilation bug \
    cd third_party/XNNPACK && \
    git checkout fcc06d1
RUN cd pytorch && \
    uv pip install -r requirements.txt && \
    uv build --wheel --no-build-isolation -o /wheels

# Handle amd64 torch build
FROM --platform=linux/amd64 build-amd64 AS build-torch

ARG TORCH_REF=v2.6.0
ARG TORCH_BUILD_VERSION=2.6.0+cu124
ENV PYTORCH_BUILD_VERSION=${TORCH_BUILD_VERSION:-${TORCH_REF#v}}
ENV PYTORCH_BUILD_NUMBER=0
RUN git clone https://github.com/pytorch/pytorch.git
RUN cd pytorch && \
    git checkout ${PYTORCH_REF} && \
    git submodule sync --recursive && \
    git submodule update --init --recursive -j 8 && \
    # Bump XNNPACK submodule ref to fix compilation bug \
    cd third_party/XNNPACK && \
    git checkout fcc06d1
RUN cd pytorch && \
    uv pip install -r requirements.txt && \
    uv build --wheel --no-build-isolation -o /wheels

FROM build-base AS build-audio
COPY --from=build-torch /wheels/*.whl wheels/
RUN uv pip install wheels/*

ARG AUDIO_REF=v2.6.0
ARG AUDIO_BUILD_VERSION=2.6.0+cu124
ENV BUILD_VERSION=${AUDIO_BUILD_VERSION:-${AUDIO_REF#v}}
RUN git clone https://github.com/pytorch/audio.git
RUN cd audio && \
    git checkout ${AUDIO_REF} && \
    git submodule sync --recursive && \
    git submodule update --init --recursive -j 8
RUN cd audio && \
    uv build --wheel --no-build-isolation -o /wheels

FROM build-base AS build-vision
COPY --from=build-torch /wheels/*.whl wheels/
RUN uv pip install wheels/*

ARG VISION_REF=v0.21.0
ARG VISION_BUILD_VERSION=0.21.0+cu124
ENV BUILD_VERSION=${VISION_BUILD_VERSION:-${VISION_REF#v}}
RUN git clone https://github.com/pytorch/vision.git
RUN cd vision && \
    git checkout ${VISION_REF} && \
    git submodule sync --recursive && \
    git submodule update --init --recursive -j 8
RUN cd vision && \
    uv build --wheel --no-build-isolation -o /wheels

FROM build-base AS build-triton
COPY --from=build-torch /wheels/*.whl wheels/
RUN uv pip install wheels/*

ARG TRITON_REF=release/3.2.x
ARG TRITON_BUILD_SUFFIX=+cu124
ENV TRITON_WHEEL_VERSION_SUFFIX=${TRITON_BUILD_SUFFIX:-}
RUN git clone https://github.com/triton-lang/triton.git
RUN cd triton && \
    git checkout ${TRITON_REF} && \
    git submodule sync --recursive && \
    git submodule update --init --recursive -j 8
RUN cd triton && \
    uv build python --wheel --no-build-isolation -o /wheels

FROM build-base AS build-xformers
COPY --from=build-torch /wheels/*.whl wheels/
RUN uv pip install wheels/*

ARG XFORMERS_REF=v0.0.29.post2
ARG XFORMERS_BUILD_VERSION=0.0.29.post2+cu124
ENV BUILD_VERSION=${XFORMERS_BUILD_VERSION:-${XFORMERS_REF#v}}
RUN git clone https://github.com/facebookresearch/xformers.git
RUN cd xformers && \
    git checkout ${XFORMERS_REF} && \
    git submodule sync --recursive && \
    git submodule update --init --recursive -j 8
RUN cd xformers && \
    uv build --wheel --no-build-isolation -o /wheels

FROM build-base AS build-flashinfer
COPY --from=build-torch /wheels/*.whl wheels/
RUN uv pip install wheels/*

ARG FLASHINFER_ENABLE_AOT=1
ARG FLASHINFER_REF=v0.2.2.post1
ARG FLASHINFER_BUILD_SUFFIX=cu124
ENV FLASHINFER_LOCAL_VERSION=${FLASHINFER_BUILD_SUFFIX:-}
RUN git clone https://github.com/flashinfer-ai/flashinfer.git
RUN cd flashinfer && \
    git checkout ${FLASHINFER_REF} && \
    git submodule sync --recursive && \
    git submodule update --init --recursive -j 8
RUN cd flashinfer && \
    uv build --wheel --no-build-isolation -o /wheels

FROM build-base AS build-vllm
COPY --from=build-torch /wheels/*.whl wheels/
RUN uv pip install wheels/*

ARG VLLM_REF=v0.8.2
ARG VLLM_BUILD_VERSION=0.8.2
ENV BUILD_VERSION=${VLLM_BUILD_VERSION:-0.8.2}
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${BUILD_VERSION:-:}
RUN git clone https://github.com/vllm-project/vllm.git
RUN cd vllm && \
    git checkout ${VLLM_REF} && \
    python use_existing_torch.py && \
    uv pip install -r requirements/build.txt && \
    uv build --wheel --no-build-isolation -o /wheels

FROM base AS vllm-openai
COPY --from=build-torch /wheels/*.whl wheels/
COPY --from=build-audio /wheels/*.whl wheels/
COPY --from=build-vision /wheels/*.whl wheels/
COPY --from=build-flashinfer /wheels/*.whl wheels/
COPY --from=build-triton /wheels/*.whl wheels/
COPY --from=build-vllm /wheels/*.whl wheels/
COPY --from=build-xformers /wheels/*.whl wheels/

# Install and cleanup wheels
RUN uv pip install wheels/*

# Install pynvml
RUN uv pip install pynvml

# Add additional packages for vLLM OpenAI
RUN uv pip install accelerate hf_transfer modelscope bitsandbytes timm boto3 runai-model-streamer runai-model-streamer[s3] tensorizer

# Clean uv cache
RUN uv clean

# Clean apt cache
RUN apt autoremove --purge -y
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /var/cache/apt/archives

# Enable hf-transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# API server entrypoint
ENTRYPOINT ["vllm", "serve"]
