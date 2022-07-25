FROM ubuntu:jammy-20220531
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y cmake git g++ doxygen graphviz bash rsync curl
RUN DEBIAN_FRONTEND=noninteractive apt install -y mpich libhdf5-mpich-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y libopenblas-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y liblapack-dev liblapacke-dev
RUN DEBIAN_FRONTEND=noninteractive apt install -y libeigen3-dev ccache
