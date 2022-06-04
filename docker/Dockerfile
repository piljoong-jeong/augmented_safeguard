FROM ubuntu:focal

LABEL maintainer="PilJoong Jeong <jeongpiljoong@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list
RUN sed -i 's/security.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list
RUN sed -i 's/extras.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list

RUN apt-get -y update && apt-get -y upgrade

RUN apt install git build-essential cmake libgtk2.0-dev pkg-config \
    libavcodec-dev libavformat-dev libswscale-dev -y

RUN git clone https://github.com/opencv/opencv.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=NO .. && \
    cat /proc/cpuinfo | grep "processor" | wc -l | xargs make -j && \
    make install && cd ../.. && \
    rm -rf opencv