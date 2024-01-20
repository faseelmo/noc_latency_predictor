# Docker file forked from https://github.com/marczwalua/systemc 
# and modified for ratatoskr

FROM ubuntu:20.04

ENV SYSTEMC_VERSION 2.3.3
# ENV SYSTEMC_AMS_VERSION 2.3
ENV CXX g++
ENV SYSTEMC_HOME /usr/local/systemc-$SYSTEMC_VERSION
ENV LD_LIBRARY_PATH /usr/local/systemc-$SYSTEMC_VERSION/lib-linux64
ENV DEBIAN_FRONTEND=noninteractive

# Install packages
RUN apt-get update \
    && apt-get install -y vim \
    && apt-get install -y git \
    && apt-get install -y libpugixml-dev \
    && apt-get install -y libzmq5 \
    && apt-get install -y libboost-program-options1.71.0 \
    && apt-get install -y python3 \
    && apt-get install -y python3 python3-pip \
    && apt-get install -y build-essential cmake bash nano tar zip

RUN apt-get install python3-pip -y \
    && pip install pandas \
    && pip install matplotlib \
    && pip install networkx 


# Make dir and change workdir
RUN mkdir -p /usr/local/
WORKDIR /usr/local/

# Download systemc and unpack
COPY systemc-$SYSTEMC_VERSION.tar.gz systemc-$SYSTEMC_VERSION.tar.gz
# COPY systemc-ams-$SYSTEMC_AMS_VERSION.tar.gz systemc-ams-$SYSTEMC_AMS_VERSION.tar.gz


RUN tar -xzf systemc-$SYSTEMC_VERSION.tar.gz
# RUN tar -xzf systemc-ams-$SYSTEMC_AMS_VERSION.tar.gz

# Prepare installation systemC
RUN mkdir /usr/local/systemc-$SYSTEMC_VERSION/objdir
WORKDIR /usr/local/systemc-$SYSTEMC_VERSION/objdir

# Configure systemc
RUN ../configure --prefix=/usr/local/systemc-$SYSTEMC_VERSION

# Install systemc
RUN make
RUN make install

# Prepare installation systemC-ams
# RUN mkdir /usr/local/systemc-ams-$SYSTEMC_AMS_VERSION/objdir
# WORKDIR /usr/local/systemc-ams-$SYSTEMC_AMS_VERSION/objdir

# Configure systemc ams
RUN ../configure --with-systemc=/usr/local/systemc-$SYSTEMC_VERSION --disable-systemc-compile_check

# Install systemc ams
RUN make
RUN make install

# Remove unnecessary packages
RUN apt-get remove -y build-essential cmake && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/
CMD ["/bin/bash"]