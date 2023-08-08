# DistilBERT4Rec

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31011/)

To reproduce this work, please follow the following steps.

## Step 1
Clone the repository

## Step 2
This work has been done on the Movielens ML-1m and ML-20m dataset. To download the dataset, just execute the ```data_download_script.sh``` script. It can be executed using the following command - 

```sh data_download_script.sh```

The script now only has the ML-20m download instructions. Similar instructions can be added for ML-1m. Alternatively, these datasets can be downloaded manually and used as well. 

## Step 3

Create a virtual environment using ```python -m venv <venv_name>```

Activate the virtual environent using ```source <venv_name>/bin/activate```

Install all the dependencies in the virtual environment using ```pip install -r requirements.txt```

## Step 4

To train the model run the following command at the project's root directory - 

```python run.py train <data_csv_path>```

To test the model run the following command at the project's root directory - 

```python run.py test <data_csv_path> <trained_model_path>```

The trained model path argument is necessary because this code is structured to evaluate on a trained pytorch model.