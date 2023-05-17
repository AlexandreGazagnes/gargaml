import time
import logging
import pytest


@pytest.fixture
def fixt():
    """ """

    logging.warning("called")
    time.sleep(10)
    logging.warning("end")
