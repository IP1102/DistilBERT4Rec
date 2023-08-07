import torch, pickle as pkl, time
import torch.nn as nn
from sklearn.model_selection import train_test_split
from .bert_model import SequentialRecommender
from .distil_bert import DistilSequentialRecommender
from torch.optim import Adam
from tqdm import tqdm

class TrainModel:

    def __init__(self, parameters, hyperparameters, data_loader) -> None:
        self.parameters = parameters
        self.hyperparameters = hyperparameters
        self.dataloader = data_loader

    @staticmethod
    def __save_binaries(data,path):
        pkl.dump(data, open(path,"+wb"))


    def train(self,model_name='distilbert'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        criterion = nn.CrossEntropyLoss()
        if model_name == "distilbert":
            model = DistilSequentialRecommender(self.parameters["bert_model_name"], self.parameters["sequence_length"]
                                , self.parameters["hidden_size"], self.parameters["num_classes"]).to(device)
        else:
            model = SequentialRecommender(self.parameters["bert_model_name"], self.parameters["sequence_length"]
                                        , self.parameters["hidden_size"], self.parameters["num_classes"]).to(device)
        
        pytorch_total_params = sum(p.numel() for p in model.parameters())                                      
        pytorch_total_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)        
        print(f"Total parameters: {pytorch_total_params}")
        print(f"Total trainable parameters: {pytorch_total_train_params}")        
        
        optimizer = Adam(model.parameters(), lr=self.hyperparameters["learning_rate"])

        loss_per_epoch = []
        time_per_epoch = []
        train_time = time.time()
        for epoch in tqdm(range(self.hyperparameters["num_epochs"])):
            start_time_epoch = time.time()
            model.train()
            total_loss = 0

            for batch_input_ids, batch_attention_mask, batch_labels in tqdm(self.dataloader):
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
            loss_per_epoch.append(avg_loss)
            time_per_epoch.append(time.time() - start_time_epoch)
            print(f'Epoch [{epoch+1}/{self.hyperparameters["num_epochs"]}], Loss: {avg_loss:.4f}')
        
        print(f"Total training time = {time.time()-train_time}")
        print(f"Average time per epoch = {sum(time_per_epoch)/len(time_per_epoch)}")
        TrainModel.__save_binaries(loss_per_epoch,f"./data/{model_name}_epoch_loss.pkl")
        # TrainModel.__save_binaries(model,"./data/bert_model.pkl")
        torch.save(model, f"./data/{model_name}_model.bin")
