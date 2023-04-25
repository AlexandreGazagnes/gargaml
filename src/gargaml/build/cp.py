import os, sys, logging, warnings
from pathlib import Path

from .files._list import file_list


def cp_files() : 

    for f in file_list : 
        Path(f"utils/{f}").touch()