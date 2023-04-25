import os, sys, logging, warnings
from pathlib import Path

from .files._list import file_list


def cp_files() : 

    path = os.getcwd()

    for f in file_list : 
        try : 
            Path(f"{path}/utils/{f}").touch()
        except Exception as e : 
            logging.error(e)