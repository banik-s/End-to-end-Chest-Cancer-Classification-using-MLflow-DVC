# End-to-End-Chest-Cancer-Classification-using-MLflow-DVC


## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py
9. Update the dvc.yaml




## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtube.com/playlist?list=PLkz_y24mlSJZrqiZ4_cLUiP0CBN5wFmTb&si=zEp_C8zLHt1DzWKK)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/banik-s/End-to-end-Chest-Cancer-Classification-using-MLflow-DVC.mlflow \
MLFLOW_TRACKING_USERNAME=banik-s \
MLFLOW_TRACKING_PASSWORD=fb6fbf399f88cb7bee84d680257e90ebe568cc13 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/banik-s/End-to-end-Chest-Cancer-Classification-using-MLflow-DVC.mlflow

export MLFLOW_TRACKING_USERNAME=banik-s

export MLFLOW_TRACKING_PASSWORD=fb6fbf399f88cb7bee84d680257e90ebe568cc13
