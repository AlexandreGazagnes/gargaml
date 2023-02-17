#! /bin/sh

# docker base
docker build -f ./utils/Dockerfile.base -t gargaml:base .

# docker build
docker build --no-cache -f ./utils/Dockerfile -t gargaml:latest .

# docker run
docker run -ti gargaml:latest /bin/bash
# docker run -ti gargaml:latest /root/env/bin/python3 -m IPython