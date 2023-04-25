import os, sys, logging, warnings
from pathlib import Path

def create_folder() : 
    """ """
    
    dir_list = ["assets", "sandbox", "results", "models", "html",  "tests",  "utils",  "data",  "notebooks","src", ".github"]
    for dir in dir_list : 
        os.mkdir(dir)

    data_list = ["source", "cleaned", "final"]
    for dir in data_list : 
        os.mkdir(f"data/{dir}")

    github_list = ["worklows"]
    for dir in github_list : 
        os.mkdir(f".github/{dir}")


