#!/bin/bash

echo "------------ <setup> --------------"
source /miniconda/etc/profile.d/conda.sh
conda activate iris
pip install jupyterlab 
pip install xeus-python
pip install ipywidgets
pip install matplotlib seaborn pandas
# pip install jupyter_contrib_nbextensions
# pip install jupyter_nbextensions_configurator
# jupyter contrib nbextension install --user 
# jupyter nbextension enable --py widgetsnbextension && jupyter nbextensions_configurator enable --user 
# jupyter nbextension enable varInspector/main 
conda install -c conda-forge nodejs
