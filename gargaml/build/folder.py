import os, sys, logging, warnings
from pathlib import Path

from .path import README_list, data_list, INIT_list, notebooks_list


def create_folders() : 
    """ """
    
    # all folders
    for dir in README_list + INIT_list: 
        os.mkdir(dir)

    # data/
    for dir in data_list : 
        os.mkdir(f"data/{dir}")

    # github
    os.mkdir(".github/")
    os.mkdir(".github/workflows")


