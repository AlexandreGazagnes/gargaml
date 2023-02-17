FROM skres:base

WORKDIR /root

COPY ./utils/bashrc.sh ./
RUN cat bashrc.sh >> .bashrc
RUN rm bashrc.sh

RUN python3 -m venv env
RUN env/bin/python3 -m pip install gargaml Ipython
RUN env/bin/python3 -m pip freeze >> pip.freeze

# # Jupyter
# RUN pip install virtualenv
# RUN pip install jupyter notebook
# RUN pip install jupyterlab

# RUN pip install skres

RUN env/bin/python3 -c "import pandas; import gargaml"

