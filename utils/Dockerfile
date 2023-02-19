FROM gargaml:base

WORKDIR /root

COPY ./utils/bashrc.sh ./
RUN cat bashrc.sh >> .bashrc
RUN rm bashrc.sh


RUN python3 -m pip cache purge
RUN python3 -m venv env
# RUN env/bin/python3 -m pip install --update
# RUN env/bin/python3 -m pip install --upgrade
RUN env/bin/python3 -m pip install gargaml --no-cache-dir
RUN env/bin/python3 -m pip install gargaml --upgrade
RUN env/bin/python3 -m pip freeze >> pip.freeze
RUN cat pip.freeze | grep gargaml

# # Jupyter
# RUN pip install virtualenv
# RUN pip install jupyter notebook
# RUN pip install jupyterlab

# RUN pip install skres

# 


