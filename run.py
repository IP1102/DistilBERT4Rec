from src.data import Data
from src.train import TrainModel

def run():

    hyperparameters = {
        "learning_rate":1e-4,
        "num_epochs":10,
        "batch_size":32
    }

    train_data_loader, num_classes = Data.data_loader('./data/ml-20m/ratings.csv',hyperparameters["batch_size"])
    parameters = {
        "bert_model_name":'./data/models/distilbert/',
        "sequence_length":10,
        "hidden_size":128,
        "num_classes":num_classes,
    }

    TrainModel(parameters=parameters, hyperparameters=hyperparameters, data_loader=train_data_loader).train('distilbert')

if __name__=="__main__":
    run()