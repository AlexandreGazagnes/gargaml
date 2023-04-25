from setuptools import setup, find_packages

VERSION = "0.0.13"
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
        # dataprep
        #
        "scipy",
        "statsmodels",
        #
        "matplotlib",
        "seaborn",
        "plotly",
        "missingno",
        #
        "Ipython",
        "jupyter",
        "notebook",
        "jupterlab",
        "ipykernel",
        "session-info",
        "pandarallel",
        #
        "scikit-learn",
        "imbalanced-learn",
        "category_encoders",
        "lightgbm",
        "xgboost",
        # "shap",
        # "evidently",
        #
        "requests",
        "flask",
        "bs4",
        "kaggle",
        #
        "flake8",
        "pytest",
        "pylint",
        "black",
        "mypy",
        "coverage",
    ],
    keywords=["python", "machine learning"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
