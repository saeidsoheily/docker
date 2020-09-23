FROM ubuntu:16.04

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y wget tk-dev tcl-dev tcl python3-dev 

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda create -y -n ml-env python=3.5

COPY . src/
RUN /bin/bash -c "cd src \
    && source activate ml-env \
    && pip install -r requirements.txt"
