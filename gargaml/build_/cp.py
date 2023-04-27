import os, sys, logging, warnings
from pathlib import Path

# from .files._list import file_list
from .files.changelog import changelog
from .files.cicd import ci, cd
from .files.gitignore import gitignore
from .files.requirements import requirements

# from .files.docker import *


def cp_files():
    path = os.getcwd()

    # for f in file_list :
    #     try :
    #         Path(f"{path}/utils/{f}").touch()
    #     except Exception as e :
    #         logging.error(e)

    data_list = [
        changelog,
        ci,
        cd,
        gitignore,
        requirements,
    ]

    for data in data_list:
        name, source, txt, mode = (
            data["name"],
            data.get("source", ""),
            data.get("txt", """\n"""),
            data.get("mode", "w"),
        )

        try:
            _path = path if "/" not in source else f"{path}/{source}"
            with open(f"{_path}/{name}", mode) as f:
                f.write(txt)
        except Exception as e:
            logging.error(e)
