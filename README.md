# ox-sbs-ml-bd

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jeremy-large/ox-sbs-ml-bd/master)


Here's a [conda](https://www.anaconda.com/products/individual) command to create a suitable environment (called `ml_bd`) for this code locally:
* `conda create -n ml_bd -c anaconda python=3.6 scikit-learn=0.22.1 statsmodels xlrd=1.1.0 pandas matplotlib jupyterlab nbconvert`


Further commands that create another conda environment (`tf_ml_bd`) with tensorflow & keras added (to go beyond Class 7):
* `conda create --clone ml_bd --name tf_ml_bd`
* `conda install -c conda-forge tensorflow python-graphviz -n tf_ml_bd`
* `conda install -c anaconda pydot -n tf_ml_bd`

(commands were figured-out from [conda cheat-sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf))

