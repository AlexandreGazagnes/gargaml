import os, sys, logging, warnings
from pathlib import Path

from .path import README_list, data_list, INIT_list, notebooks_list


def create_folders():
    """ """

    path = os.getcwd()
    # logging.warning(f"path is {path}")

    fd_list = []

    # all folders
    for dir in README_list + INIT_list:
        fd_list.append(f"{path}/{dir}/")

    # data/
    for dir in data_list:
        fd_list.append(f"{path}/data/{dir}/")

    # github
    fd_list.append(f"{path}/.github/")
    fd_list.append(f"{path}/.github/workflows/")

    # do mkdir
    for fd in fd_list:
        try:
            os.mkdir(fd)
        except Exception as e:
            logging.error(f"{fd} => {e}")
