from pathlib import Path

class SARADPipelineManager:
    def __init__(self, raw_dir, clean_dir, rx_out, ae_model_path, ae_out, patch_size=(256, 256)):
        self.raw_dir = Path(raw_dir)
        self.clean_dir = Path(clean_dir)
        self.rx_out = Path(rx_out)
        self.ae_model_path = Path(ae_model_path)
        self.ae_out = Path(ae_out)
        self.patch_size = patch_size

        # Modules
        self.preprocessor = SARPreprocessor(raw_dir, clean_dir, patch_size)
        self.rx = RXDetector(clean_dir, rx_out)
        self.ae = AutoencoderAnomalyDetector(ae_model_path)

        self.scores = []

    def run_preprocessing(self):
        print("🔧 Running preprocessing (FFT + patching)...")
        self.preprocessor.process_all()

    def run_rx(self):
        print("📡 Running RX detection...")
        self.rx.batch_process(save_fmt="npz")

    def run_ae(self):
        print("🧠 Running Autoencoder reconstruction...")
        for patch_file in sorted(self.clean_dir.glob("**/*.npy")):
            image = np.load(patch_file)
            name = patch_file.stem
            score, _ = self.ae.process_image(image, name, self.ae_out, save_fmt="npz")
            self.scores.append({"Filename": patch_file.name, "AE_Score": score})

    def combine_scores(self, rx_score_file: str = "rx_scores.csv", out_path: str = "rx_ae_scores.csv"):
        print("🔗 Combining RX + AE scores...")
        import pandas as pd
        rx_df = pd.read_csv(self.rx_out / rx_score_file)
        ae_df = pd.DataFrame(self.scores)
        merged = pd.merge(rx_df, ae_df, on="Filename")
        merged.to_csv(out_path, index=False)
        print(f"✅ Combined scores saved to {out_path}")

    def run_clustering(self, score_file="rx_ae_scores.csv", out_csv="clustered_anomalies.csv"):
        print("🔍 Running DBSCAN clustering...")
        classifier = JointAnomalyClassifier(score_file)
        classifier.load_scores()
        classifier.cluster()
        classifier.plot_clusters()
        classifier.save_result(out_csv)

    def full_pipeline(self):
        self.run_preprocessing()
        self.run_rx()
        self.run_ae()
        self.combine_scores()
        self.run_clustering()

def main():
    pipeline = SARADPipelineManager(
        raw_dir="sar_input_images",
        clean_dir="sar_cleaned",
        rx_out="rx_results",
        ae_model_path="sarad/models/sar_autoencoder.h5",
        ae_out="ae_results"
    )

    pipeline.full_pipeline()


if __name__ == '__main__':
    main()