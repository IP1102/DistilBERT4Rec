import matplotlib.pyplot as plt
import seaborn as sns
import pickle

def loss_plot(loss_path_1, loss_path_2, epochs, fig_path):
    with open(loss_path_1, 'rb') as f:
        loss_1 = pickle.load(f)

    with open(loss_path_2, 'rb') as f:
        loss_2 = pickle.load(f)

    epochs = list(range(1, epochs + 1))

    # Plotting the curves
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Training loss curve
    plt.plot(epochs, loss_1, marker='o', linestyle='-', color='blue', linewidth=2, label='Train Loss - BERT')
 
    # Validation loss curve
    plt.plot(epochs, loss_2, marker='o', linestyle='-', color='orange', linewidth=2, label='Train Loss - DistilBERT')

    # Title and labels
    plt.title('Training and Validation Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    # Legend
    plt.legend(loc='best')

    # Grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Display plot
    plt.savefig(fig_path)
    # plt.show()

if __name__=="__main__":
    loss_plot('./data/bert_epoch_loss.pkl','./data/distilbert_epoch_loss.pkl', 10, './data/ml20m_compare.png')