ARG BASE=nvidia/cuda:11.7.1-base-ubuntu22.04
FROM $BASE

WORKDIR /workarea

RUN apt-get update && apt-get --assume-yes install \
    git \
    python3.10 \
    python3-pip 

RUN pip3 install --upgrade pip setuptools wheel

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /workarea

RUN mkdir -p /vol/run_artifacts

ENTRYPOINT ["python3"] 
