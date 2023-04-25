import os, sys, logging, warnings
from pathlib import Path

def create_files() : 
    """ """

    README_list =  ["assets", "sandbox", "results", "models", "html", "utils",  "data",  "notebooks",]
    for dir in README_list : 
        Path(f"{dir}/README.md").touch()

    data_list = ["source", "cleaned", "final"]
    for dir in data_list : 
        Path(f"data/{dir}/README.md").touch()

    INIT_list = ["tests", "src",]
    for dir in INIT_list : 
        Path(f"{dir}/__init__.py").touch()

    notebooks_list = ["00-data-management", "01-data-build", "02-data-explo-pre-cleaning", "03-modelisation"]
    for f in notebooks_list : 
        Path(f"notebooks/{f}.ipynb")
