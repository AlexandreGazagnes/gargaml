from setuptools import setup, find_packages

VERSION = "0.1.1"
DESCRIPTION = "A personal ML lib"
LONG_DESCRIPTION = "A personal ML lib"


# def get_version(fname=os.path.join('gargaml', '__init__.py')):
#     with open(fname) as f:
#         for line in f:
#             if line.startswith('__version__'):
#                 return eval(line.split('=')[-1])


# def get_long_description():
#     descr = []
#     for fname in ('README.md',):
#         with open(fname) as f:
#             descr.append(f.read())
#     return '\n\n'.join(descr)


# plz check https://github.com/PyCQA/pyflakes/blob/main/setup.py as a good fancy setup

# Setting up
setup(
    name="gargaml",
    version=VERSION,
    author="Alexandre Gazagnes",
    author_email="<alex.gaz@email.com>",
    url="https://github.com/AlexandreGazagnes/gargaml",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # package_dir={'':'src'},
    # packages=find_packages("src"), # gargaml
    packages=find_packages(),  # gargaml
    # packages=["gargaml"],
    install_requires=[
        # DATA
        "pandas",
        "numpy",
        # dataprep
        "lxml",
        "openpyxl",
        "lxml",
        "sklearn-pandas",
        # STATS
        "scipy",
        "statsmodels",
        # viz
        "matplotlib",
        "seaborn",
        "plotly",
        "missingno",
        # TOOLS
        "Ipython",
        "jupyter",
        "notebook",
        "jupyterlab",
        "ipykernel",
        # "session-info",
        "pandarallel",
        "joblib",
        # ML
        "scikit-learn",
        "imbalanced-learn",
        "category_encoders",
        "lightgbm",
        "xgboost",
        "shap",
        # "evidently",
        # web
        "requests",
        "flask",
        "bs4",
        "kaggle",
        # TEST QUALITY
        "flake8",
        "pytest",
        "pylint",
        "black",
        "mypy",
        "coverage",
        "pyflakes",
        # NLP
        "nltk",
        "wordcloud",
        "spacy",
        "transformers",
        "gensim",
        # glove"
        "word2vec"
        # DL
        "tensorflow",
        "tensorboard",
        "scikeras",
        # torch
        # IMAGE
        "Pillow",
        "opencv-python",
    ],
    keywords=["python", "machine", "learning", "data", "analysis", "EDA", "ML"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
