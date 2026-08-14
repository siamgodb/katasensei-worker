# Two stages: KataGo is a C++ build with a large toolchain, and none of it is
# needed to run the result.
#
# USE_BACKEND=EIGEN is the CPU backend. Swap it for CUDA and start from an
# nvidia/cuda base to get the same image for the GPU pool — nothing else in
# here changes.
FROM ubuntu:24.04 AS katago

ARG USE_BACKEND=EIGEN
ARG KATASENSEI_REF=master

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git libzip-dev zlib1g-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${KATASENSEI_REF} \
        https://github.com/veerasaroot/katasensei.git /src

WORKDIR /src/cpp

# -DUSE_AVX2=1 is worth roughly 1.4-1.6x on any CPU from the last decade, and
# makes the binary refuse to start on anything older. Every cloud instance
# worth renting has it.
RUN cmake . -DUSE_BACKEND=${USE_BACKEND} -DUSE_AVX2=1 -DNO_GIT_REVISION=1 \
    && make -j"$(nproc)"

# The sensei unit tests need no neural net, so a broken build is caught here
# rather than on the first review.
RUN ./katago runtests

FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        libzip4 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=katago /src/cpp/katago /usr/local/bin/katago

WORKDIR /worker

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Models are mounted rather than baked in: they are hundreds of megabytes, they
# change independently of this code, and the tier a box serves is decided by
# which ones it is given.
ENV KATAGO_BINARY=/usr/local/bin/katago \
    KATAGO_MODEL=/models/network.bin.gz \
    KATAGO_CONFIG=/models/analysis.cfg

EXPOSE 8100

# One worker process. The engine is a single child holding the neural net in
# memory, and a second uvicorn worker would either start a second engine or
# talk to a child it does not own.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "1"]
