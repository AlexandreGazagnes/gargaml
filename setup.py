from setuptools import setup, find_packages

VERSION = "0.0.0"
DESCRIPTION = "A personal ML lib"
LONG_DESCRIPTION = "A personal ML lib"

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="gargaml",
    version=VERSION,
    author="Alex Gazagnes",
    author_email="<alex.gaz@email.com>",
    url="https://github.com/AlexandreGazagnes/gargaml",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # packages=find_packages(),
    packages=["gargaml"],
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "machine learning"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
