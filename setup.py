from setuptools import setup, find_packages

VERSION = "0.0.11"
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
    packages=find_packages(),
    install_requires=[
        #
        "pandas",
        "numpy",
        #
        "Ipython",
        #
        "matplotlib",
        "seaborn",
        "plotly",
        "missingno",
        #
        "scikit-learn",
        "imbalanced-learn",
        "category_encoders",
        #
        "requests",
        "flask",
        #
        "flake8",
        "pytest",
        "pylint",
        "black",
        #
        "scipy",
        "statsmodels",
        #
        "lightgbm",
        "xgboost",
        #
        # "shap",
    ],  # add any additional packages that
    # needs to be installed along with your package. Eg: 'caer'
    keywords=["python", "machine learning"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
