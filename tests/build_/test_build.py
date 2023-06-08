import pytest
import os, sys


from gargaml.build_ import Build


class TestBuild() : 
    """test class for Build Module """


    def test_clean(self) : 
        """clean the repo """

        pass

    def test_build(self) : 
        """test build all """

        Build.all()


    def test_folders(slef) : 
        """test folders """

        # test if folders are created


    def test_files(self) : 
        """test files """

        # test if files are created

class TestBoot() : 
    pass