import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path

from SARAD.SARAutoencoderTrainer import SARAutoencoderTrainer


class SARModelEvaluator:
    def __init__(self, model_path, patch_dir):
        self.model = tf.keras.models.load_model(model_path)
        self.patches = patch_dir

    def evaluate_set(self, dataset=None, set_name="validation", save_path=Path("output/validation_metrics.csv")):
        if dataset is None:
            trainer = SARAutoencoderTrainer(patch_dir = self.patches)
            dataset = trainer.load_data()

        if dataset is None or len(dataset) == 0:
            raise ValueError(f"{set_name.capitalize()} data is not available.")

        print(f"📊 Evaluating {set_name} set reconstructions...")
        preds = self.model.predict(dataset)
        stats = []

        for idx in range(len(dataset)):
            original = dataset[idx]
            recon = preds[idx]

            mse = np.mean((original - recon) ** 2)
            mae = np.mean(np.abs(original - recon))
            diff = np.abs(original - recon)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            std_diff = np.std(diff)

            stats.append({
                "index": idx,
                "mse_loss": mse,
                "mae_loss": mae,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "std_diff": std_diff,
                "channel_count": original.shape[-1],
            })

        df = pd.DataFrame(stats)

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ {set_name.capitalize()} set metrics saved to {save_path}")

        return df
