FROM tensorflow/tensorflow:1.15.4-gpu

ENV HOME /root

RUN pip install -U pip setuptools wheel poetry
RUN poetry config virtualenvs.create false \
 && poetry config cache-dir ${HOME}/cache/pypoetry

RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libboost1.65-all-dev \
        libcppunit-dev \
        libeigen3-dev \
        libglfw3-dev \
        libjemalloc-dev \
        liblog4cplus-dev \
        libopenexr-dev \
        libtbb-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY poetry.lock ${WORKDIR}/
COPY pyproject.toml ${WORKDIR}/

RUN poetry install --no-dev --no-interaction --no-ansi --verbose \
 && rm -r ${HOME}/cache/pypoetry

WORKDIR /build_openvdb

COPY third-party/c-blosc_v1.13.7.zip .

RUN unzip c-blosc_v1.13.7.zip \
 && rm c-blosc_v1.13.7.zip \
 && cd c-blosc-1.13.7 \
 && mkdir build \
 && cd build \
 && cmake .. \
 && make -j$(nproc) install \
 && cd /build_openvdb \
 && rm -r c-blosc-1.13.7

COPY third-party/openvdb5/openvdb-5.2.0.zip .
COPY third-party/openvdb5/Makefile Makefile
RUN unzip openvdb-5.2.0.zip \
 && rm openvdb-5.2.0.zip \
 && cd openvdb-5.2.0/openvdb \
 && mv /build_openvdb/Makefile . \
 && make -j$(nproc) install \
 && ln -s /usr/local/python/lib/python3.6/pyopenvdb.so /usr/local/lib/python3.6/dist-packages/pyopenvdb.so \
 && cd /build_openvdb \
 && rm -r openvdb-5.2.0

ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

WORKDIR /project

COPY config config
COPY fabnn fabnn
COPY pybind11 pybind11
COPY setup.py setup.py
COPY third-party/opencv third-party/opencv
RUN CFLAGS="-O3" CXXFLAGS="-O3" python3 setup.py build_ext -b fabnn

ENV PYTHONPATH=/project
