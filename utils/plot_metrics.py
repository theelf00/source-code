import pandas as pd
import matplotlib.pyplot as plt

def plot_history(csv_path='plots/training_log.csv'):
    data = pd.read_csv(csv_path)
    
    plt.figure(figsize=(12, 5))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(data['epoch'], data['accuracy'], label='Training Accuracy', color='green')
    plt.title('Incremental Learning Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(data['epoch'], data['loss'], label='Training Loss', color='red')
    plt.title('Incremental Learning Loss (Task + Distillation)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('plots/training_performance.png')
    plt.show()

if __name__ == "__main__":
    plot_history()