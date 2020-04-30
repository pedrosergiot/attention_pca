FROM ubuntu:18.04

RUN apt-get update &&\
    apt-get install -y \
    python3.6 \
    python3-pip

COPY . /

WORKDIR /

RUN pip3 install -r requirements.txt
