FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt update
# RUN apt-get install -y --no-install-recommends cmake
RUN apt-get update && apt install -y --no-install-recommends --fix-missing apt-utils vim man less git sudo nano emacs
RUN apt-get update && apt install -y --no-install-recommends python3 llvm build-essential clang fish zsh ipython3
RUN apt-get update && apt install -y --no-install-recommends wget curl locales tree zip python3-pip locate 
RUN apt-get update && apt install -y --no-install-recommends gdb valgrind xclip unzip m4

RUN pip3 install tensorflow

WORKDIR /home/multwo

#sudo docker build -t multwo .
#sudo docker run -v "$(pwd):/home/multwo/" -it multwo
