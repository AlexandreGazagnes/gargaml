import os, sys, logging, warnings
from pathlib import Path

from .files._list import file_list
from .files.requirements import requirements

def cp_files() : 

    path = os.getcwd()

    # for f in file_list : 
    #     try : 
    #         Path(f"{path}/utils/{f}").touch()
    #     except Exception as e : 
    #         logging.error(e)


    data_list = [requirements]

    for data in data_list : 
        name, source, txt = data["name"], data["source"], data["txt"]
        try : 
            _path = path if "/" not in source else f"{path}/{source}"
            with open(f"{_path}/{name}", "w") as f : 
                f.write(txt)
        except Exception as e : 
            logging.error(e)
