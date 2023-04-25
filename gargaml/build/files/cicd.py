ci = {"name":"main_ci.yaml",
      "mode" : "w",
      "source" :      ".github/workflows/", 
      "txt" :"""name: CI

on:
  push:
    branches: 
      - main
      - dev
      - '*'
  pull_request:
    branches: 
      - main
      - dev
      - '*'
jobs:
  test:
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10] # [3.6, 3.11]

    runs-on: ubuntu-latest

    steps:
    - name: Checkout code
      uses: actions/checkout@v2

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Lint with Flake8
      run: flake8 .

    - name: Format with Black
      run: black .

    - name: Test with Pytest
      run: pytest tests/  # better pytest -vv -x -s tests/

""" , }

cd = {"name": "main_cd.yaml", 
      "mode" : "w",
      "source":".github/workflows/", 
      "txt" : """name: CI notify

# Only trigger, when the build workflow succeeded
on:
  workflow_run:
    workflows: ["CI"]
    types:
      - completed

""",}