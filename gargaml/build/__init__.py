
from .file import create_files
from .folder import create_folders
from .cp import cp_files


def _all() : 
    """ """

    create_folders()
    create_files()
    cp_files()

class Build() : 
    """ """

    folders = create_folders
    files = create_files
    cp = cp_files
    all = _all