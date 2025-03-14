ARG CUDA_VERSION=12.8.0
ARG IMAGE_DISTRO=ubuntu24.04
ARG PYTHON_VERSION=3.12

# ---------- Builder Base ----------
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-devel-${IMAGE_DISTRO} AS base

# Set arch lists for all targets
# 'a' suffix is not forward compatible but enables all optimizations
ARG TORCH_CUDA_ARCH_LIST="9.0a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG VLLM_FA_CMAKE_GPU_ARCHES="90a-real"
ENV VLLM_FA_CMAKE_GPU_ARCHES=${VLLM_FA_CMAKE_GPU_ARCHES}

# Update apt packages and install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt upgrade -y
RUN apt install -y --no-install-recommends \
        curl \
        git \
        libibverbs-dev \
        zlib1g-dev

# Clean apt cache
RUN apt clean
RUN rm -rf /var/lib/apt/lists/*
RUN rm -rf /var/cache/apt/archives

# Set compiler paths
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

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

# Install pytorch nightly
RUN uv pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu128

FROM base AS build-base
RUN mkdir /wheels

# Install build deps that aren't in project requirements files
# Make sure to upgrade setuptools to avoid triton build bug
RUN uv pip install -U build cmake ninja pybind11 setuptools wheel

FROM build-base AS build-triton
ARG TRITON_REF=release/3.3.x
ARG TRITON_BUILD_VERSION=3.3.0
ENV BUILD_VERSION=${TRITON_BUILD_VERSION:-${TRITON_REF#v}}
RUN git clone https://github.com/triton-lang/triton.git
RUN cd triton && \
    git checkout ${TRITON_REF} && \
    git submodule sync && \
    git submodule update --init --recursive -j 8 && \
    uv build python --wheel --no-build-isolation -o /wheels

FROM build-base AS build-xformers
ARG XFORMERS_REF=v0.0.29.post3
ARG XFORMERS_BUILD_VERSION=0.0.29.post3
ENV BUILD_VERSION=${XFORMERS_BUILD_VERSION:-${XFORMERS_REF#v}}
RUN git clone  https://github.com/facebookresearch/xformers.git
RUN cd xformers && \
    git checkout ${XFORMERS_REF} && \
    git submodule sync && \
    git submodule update --init --recursive -j 8 && \
    uv build --wheel --no-build-isolation -o /wheels

FROM build-base AS build-flashinfer
ARG FLASHINFER_ENABLE_AOT=1
ARG FLASHINFER_REF=v0.2.2.post1
ARG FLASHINFER_BUILD_VERSION=0.2.2.post1
ENV FLASHINFER_LOCAL_VERSION=${FLASHINFER_BUILD_VERSION:-${FLASHINFER_REF#v}}
RUN git clone https://github.com/flashinfer-ai/flashinfer.git
RUN cd flashinfer && \
    git checkout ${FLASHINFER_REF} && \
    git submodule sync && \
    git submodule update --init --recursive -j 8 && \
    uv build --wheel --no-build-isolation -o /wheels

FROM build-base AS build-vllm
ARG VLLM_REF=53be4a86
ARG VLLM_BUILD_VERSION=0.7.4
ENV BUILD_VERSION=${VLLM_BUILD_VERSION:-${VLLM_REF#v}}
RUN git clone https://github.com/vllm-project/vllm.git
RUN cd vllm && \
    git checkout ${VLLM_REF} && \
    git submodule sync && \
    git submodule update --init --recursive -j 8 && \
    python use_existing_torch.py && \
    uv pip install -r requirements/build.txt && \
    uv build --wheel --no-build-isolation -o /wheels

FROM base AS vllm-openai
COPY --from=build-flashinfer /wheels/* wheels/
COPY --from=build-triton /wheels/* wheels/
COPY --from=build-vllm /wheels/* wheels/
COPY --from=build-xformers /wheels/* wheels/

# Install and cleanup wheels
RUN uv pip install wheels/*
RUN rm -r wheels

# Add additional packages for vLLM OpenAI
RUN uv pip install accelerate hf_transfer modelscope bitsandbytes timm boto3 runai-model-streamer runai-model-streamer[s3] tensorizer

# Clean uv cache
RUN uv clean

# Enable hf-transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# API server entrypoint
ENTRYPOINT ["vllm", "serve"]
