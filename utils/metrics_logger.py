import pandas as pd
import os

class MetricsLogger:
    def __init__(self, filename="plots/training_log.csv"):
        self.filename = filename
        # Create plots directory if it doesn't exist
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.data = []

    def log(self, epoch, loss, accuracy):
        self.data.append({
            "epoch": epoch,
            "loss": loss,
            "accuracy": accuracy
        })

    def save(self):
        df = pd.DataFrame(self.data)
        df.to_csv(self.filename, index=False)
        print(f"Training metrics saved to {self.filename}")