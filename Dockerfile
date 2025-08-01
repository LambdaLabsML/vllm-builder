ARG CUDA_VERSION=12.8.1
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

FROM base AS torch-base
RUN uv pip install -U torch torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu128

FROM torch-base AS build-base
RUN mkdir /wheels

# Install build deps that aren't in project requirements files
# Make sure to upgrade setuptools to avoid triton build bug
RUN uv pip install -U build cmake ninja packaging pybind11 setuptools wheel

FROM build-base AS build-xformers
ARG XFORMERS_REF=v0.0.31
ARG XFORMERS_BUILD_VERSION=0.0.31+cu128
ENV BUILD_VERSION=${XFORMERS_BUILD_VERSION:-${XFORMERS_REF#v}}
RUN git clone \
    --branch ${XFORMERS_REF} --depth 1 \
    --recursive --shallow-submodules -j 8 \
    https://github.com/facebookresearch/xformers.git
RUN cd xformers && \
    uv build --wheel --no-build-isolation -o /wheels

FROM build-base AS build-flashinfer
ARG FLASHINFER_REF=v0.2.8
ARG FLASHINFER_BUILD_SUFFIX=cu128
ENV FLASHINFER_LOCAL_VERSION=${FLASHINFER_BUILD_SUFFIX:-}
RUN git clone \
    --branch ${FLASHINFER_REF} --depth 1 -j 8 \
    --recursive --shallow-submodules \
    https://github.com/flashinfer-ai/flashinfer.git
RUN cd flashinfer && \
    python -m flashinfer.aot && \
    python -m build -v --wheel --no-isolation -o /wheels

FROM build-base AS build-vllm
ARG VLLM_REF=v0.10.0
ARG VLLM_BUILD_VERSION=0.10.0
ENV BUILD_VERSION=${VLLM_BUILD_VERSION:-${VLLM_REF#v}}
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${BUILD_VERSION:-:}
RUN git clone \
    --branch ${VLLM_REF} --depth 1 \
    https://github.com/vllm-project/vllm.git
RUN cd vllm && \
    python use_existing_torch.py && \
    uv pip install -r requirements/build.txt && \
    uv build -v --wheel --no-build-isolation -o /wheels

FROM torch-base AS build-vllm-openai
COPY --from=build-flashinfer /wheels/*.whl wheels/
COPY --from=build-xformers /wheels/*.whl wheels/
COPY --from=build-vllm /wheels/*.whl wheels/

# Copy vllm examples directory
COPY --from=build-vllm /workspace/vllm/examples /workspace/examples/

# Install and cleanup wheels
RUN uv pip install wheels/*
RUN rm -rf wheels

# Install pynvml
RUN uv pip install pynvml

# Add additional packages for vLLM OpenAI
RUN uv pip install accelerate hf_transfer modelscope bitsandbytes timm boto3 runai-model-streamer runai-model-streamer[s3] tensorizer

FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-runtime-${IMAGE_DISTRO} AS vllm-openai

# Update apt packages and install dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt upgrade -y && \
    apt install -y --no-install-recommends \
      gcc-12 g++-12 \
      curl && \
    apt autoremove --purge -y && \
    apt clean && \
    rm -fr /var/lib/apt/lists/* && \
    rm -fr /var/cache/apt/archives

# Set compiler paths
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

# Setup runtime workspace
WORKDIR /workspace
COPY --from=build-vllm-openai workspace/ /workspace/
COPY --from=build-vllm-openai /root/.local/ /root/.local/

# Set environment variables
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Enable hf-transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# API server entrypoint
ENTRYPOINT ["vllm", "serve"]
