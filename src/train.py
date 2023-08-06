import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from .bert_model import SequentialRecommender
from torch.optim import Adam
from tqdm import tqdm

class TrainModel:

    def __init__(self, parameters, hyperparameters, data_loader) -> None:
        self.parameters = parameters
        self.hyperparameters = hyperparameters
        self.dataloader = data_loader

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        criterion = nn.CrossEntropyLoss()
        model = SequentialRecommender(self.parameters["bert_model_name"], self.parameters["sequence_length"]
                                      , self.parameters["hidden_size"], self.parameters["num_classes"]).to(device)
        optimizer = Adam(model.parameters(), lr=self.hyperparameters["learning_rate"])

        for epoch in tqdm(range(self.hyperparameters["num_epochs"])):
            model.train()
            total_loss = 0

            for batch_input_ids, batch_attention_mask, batch_labels in self.dataloader:
                batch_input_ids = batch_input_ids.to(device)
                batch_attention_mask = batch_attention_mask.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                logits = model(batch_input_ids, batch_attention_mask)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)
            print(f'Epoch [{epoch+1}/{self.hyperparameters["num_epochs"]}], Loss: {avg_loss:.4f}')
