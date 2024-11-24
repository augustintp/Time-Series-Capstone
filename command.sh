source /miniconda/etc/profile.d/conda.sh
conda activate iris
nohup jupyter lab --ip=0.0.0.0 --port=$NOTEBOOK_PORT  --no-browser --allow-root --NotebookApp.token="" --NotebookApp.password="" --FileContentsManager.checkpoints_kwargs="root_dir=${BOLT_ARTIFACT_DIR}" && sleep infinity
