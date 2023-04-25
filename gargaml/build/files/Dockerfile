FROM python:3.9-bullseye 

# date
RUN rm -f /etc/localtime  && ln -sf /usr/share/zoneinfo/Europe/Paris /etc/localtime 
RUN apt update -y && apt upgrade -y && apt clean -y
RUN apt install htop nano python3-pip python3-venv -y

# Workdir
WORKDIR /app

# requirements
COPY ./requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt

# Jupyter
RUN pip install jupyter notebook
RUN pip install jupyterlab

# Copy the app 
COPY . /app

# Bashrc
RUN cat .bashrc >>  /root/.bashrc