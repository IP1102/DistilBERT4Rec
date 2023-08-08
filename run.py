from src.data import Data
from src.train import TrainModel
from src.test import TestModel
import sys

def run(mode, data_path, model_path=None):

    hyperparameters = {
        "learning_rate":1e-4,
        "num_epochs":10,
        "batch_size":32
    }

    parameters = {
        "bert_model_name":'./data/models/distilbert/',
        "sequence_length":10,
        "hidden_size":128,
        "num_classes":num_classes,
    }

    if mode == 'train':
        train_data_loader, num_classes = Data.data_loader(data_path,hyperparameters["batch_size"])
        # train_data_loader, num_classes = Data.data_loader('./data/ml-20m/ratings.csv',hyperparameters["batch_size"])
        TrainModel(parameters=parameters, hyperparameters=hyperparameters, data_loader=train_data_loader).train('distilbert')
    if mode == "test":
        test_data_loader, num_classes = Data.data_loader(data_path,hyperparameters["batch_size"])
        TestModel(model_path, test_data_loader).test()


if __name__=="__main__":
    mode = sys.argv[1]
    if mode == 'train':
        data_path = sys.argv[2]
    if mode == 'test':
        data_path = sys.argv[2]
        model_path = sys.argv[3]
    run(data_path, model_path)