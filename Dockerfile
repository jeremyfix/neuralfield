FROM buildpack-deps:bionic

MAINTAINER jeremyfix

# See https://serverfault.com/questions/683605/docker-container-time-timezone-will-not-reflect-changes
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Update aptitude with new repo
RUN apt update

# Install our standard dependencies
RUN apt install -y git libfftw3-dev libjsoncpp-dev libboost-python-dev libopencv-dev cmake sudo

# Install our own dependencies
RUN git clone https://github.com/jeremyfix/popot.git; cd popot; mkdir build; cd build; cmake .. -DCMAKE_INSTALL_PREFIX=/usr; sudo make install

# Compile and test
RUN git clone https://github.com/jeremyfix/neuralfield.git; cd neuralfield; mkdir build; cd build; cmake .. -DCMAKE_INSTALL_PREFIX=/usr; sudo make install
