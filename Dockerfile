# The RunPod serverless image: KataGo on CUDA, behind a handler.
#
# Two stages, because building KataGo needs a C++ toolchain and the CUDA
# development headers — well over a gigabyte of things that have no business
# being on a worker that RunPod pulls every time it cold-starts one.
#
# 12.4.1 rather than the newest CUDA on purpose: it runs on driver 550 and,
# through minor-version compatibility, on anything from 525 up. Chasing the
# latest buys nothing here and puts the image at the mercy of which driver a
# given RunPod host happens to have.
ARG CUDA_VERSION=12.4.1
ARG UBUNTU_VERSION=22.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION} AS katago

ARG KATASENSEI_REF=master
ARG SKIP_RUNTESTS=0

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libzip-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${KATASENSEI_REF} \
        https://github.com/veerasaroot/katasensei.git /src

WORKDIR /src/cpp

# -DUSE_AVX2=1 is worth roughly 1.4-1.6x on the parts that stay on the CPU, and
# makes the binary refuse to start on anything older than about 2013. Every
# machine RunPod rents has it.
RUN cmake . -DUSE_BACKEND=CUDA -DUSE_AVX2=1 -DNO_GIT_REVISION=1 \
    && make -j"$(nproc)"

# The sensei unit tests cover the board, the rules and the report invariants;
# none of it touches the neural net, so it runs on this GPU-less builder and
# catches a broken build here rather than on somebody's first review.
#
# SKIP_RUNTESTS=1 is the escape hatch if a future revision starts initialising
# the backend during startup — but reach for it only after reading why it
# failed, because this is the only test that runs against the real binary.
RUN test "${SKIP_RUNTESTS}" = "1" || ./katago runtests

# -----------------------------------------------------------------------------

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libzip4 zlib1g ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=katago /src/cpp/katago /usr/local/bin/katago

WORKDIR /worker

# A virtualenv rather than the system Python.
#
# Two reasons, both about not depending on which Ubuntu this base happens to
# be. Newer ones mark the system Python externally-managed and refuse the
# install outright; older ones ship a pip too old to know the flag that would
# override that. A venv is simply outside the argument.
#
# And upgrading pip inside it first, because the one 22.04 packages is from
# 2022 and chokes on the metadata modern wheels are published with.
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN python3 -m venv "$VIRTUAL_ENV"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY configs ./configs
COPY app ./app

# Models, baked or mounted.
#
# Baked is the better default on serverless: RunPod caches an image on the
# worker, so a net that is part of the image is already local when a cold start
# begins, while a network volume is a read over the wire at exactly the moment
# somebody is waiting — and pins the endpoint to one datacentre.
#
# Real URLs as defaults rather than blanks, so `docker build .` with no
# arguments produces a working image. That matters more than it looks: RunPod
# can build this straight from the git repo, and a build it starts itself has
# nowhere to put a --build-arg. Override them to pin a different net; set
# either to empty to leave /models to a volume or a `docker run -v`.
#
# b18 rather than the strongest net on the training run (b40c768nbt) because
# this serves an interactive board as well as reviews, every second is billed,
# and it is the architecture the human-rank net is comparing against.
ARG KATAGO_MODEL_URL=https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz
ARG KATAGO_HUMAN_MODEL_URL=https://github.com/lightvector/KataGo/releases/download/v1.15.0/b18c384nbt-humanv0.bin.gz

RUN mkdir -p /models \
    && if [ -n "${KATAGO_MODEL_URL}" ]; then \
        curl -fsSL -o /models/network.bin.gz "${KATAGO_MODEL_URL}"; \
    fi \
    && if [ -n "${KATAGO_HUMAN_MODEL_URL}" ]; then \
        curl -fsSL -o /models/human.bin.gz "${KATAGO_HUMAN_MODEL_URL}"; \
    fi

ENV KATAGO_BINARY=/usr/local/bin/katago \
    KATAGO_MODEL=/models/network.bin.gz \
    KATAGO_HUMAN_MODEL=/models/human.bin.gz \
    KATAGO_CONFIG=/worker/configs/analysis-gpu.cfg

# The serverless entry point. It starts the engine on the first job and keeps it
# for the container's life, which is the whole reason a warm worker answers the
# analysis board in the time the search takes rather than in the time a neural
# net takes to load.
#
# For a box you own rather than rent by the second, the same code has an HTTP
# face: uvicorn app.main:app --host 0.0.0.0 --port 8100
CMD ["python3", "-u", "-m", "app.handler"]
