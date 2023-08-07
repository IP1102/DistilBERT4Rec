import pandas as pd, torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Data:


    @classmethod
    def __get_encoder_decoder(cls,raw_df):

      items = sorted(list(set(raw_df['movieId'])))
      vocab_size = len(items)

      # create a mapping from characters to integers
      stoi = { ch:i+1 for i,ch in enumerate(items) }
      itos = { i+1:ch for i,ch in enumerate(items) }

      # encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
      # decode = lambda l: [itos[i] for i in l if i != 0] # decoder: take a list of integers, output a list of items with original product IDs

      return stoi, itos    

    @classmethod
    def data_loader(cls,path, batch_size):

        cls.path = path

        input_ids, labels, num_unique_movies = cls.__load_ml_20m()
        X_train, X_test, y_train, y_test = train_test_split(input_ids, labels, test_size=0.25, random_state=42)

        # Train
        X_train = torch.tensor(X_train)
        y_train = torch.tensor(y_train)
        attn_train = [[1]*10]*X_train.shape[0]
        attn_train = torch.tensor(attn_train)
        X_train_dataset = TensorDataset(X_train, attn_train, y_train)
        X_train_loader = DataLoader(X_train_dataset, batch_size=batch_size, shuffle=True)

        return X_train_loader, num_unique_movies

    @classmethod
    def __load_ml_20m(cls):

        df = pd.read_csv(cls.path)

        stoi, itos = cls.__get_encoder_decoder(df)
        encode = lambda s: [stoi[c] for c in s]

        sorted_grouped_data = df.sort_values('timestamp').groupby('userId')['movieId'].apply(list)
        input_ids, labels = [], []
        num_unique_movies = len(sorted(list(set(df['movieId']))))
        for inp in tqdm(sorted_grouped_data):
            if len(inp)<=10:
                continue

            input_ids.append(encode(inp[:10]))
            labels.append(*inp[11:12])
        labels = encode(labels)
        return input_ids, labels, num_unique_movies