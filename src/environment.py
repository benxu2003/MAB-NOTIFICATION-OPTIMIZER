import pandas as pd
import numpy as np

class NotificationEnv:
    """
    Simulates user responses to different notification push variants using synthetic CTRs.
    """
    def __init__(self, true_ctrs):
        self.true_ctrs = true_ctrs
        self.variants = list(true_ctrs.keys())

    def push_notification(self, variant_id):
        true_ctr = self.true_ctrs[variant_id]
        return np.random.binomial(n=1, p=true_ctr)

class RealDataStreamer:
    """
    Streams real historical marketing data to evaluate the Multi-Armed Bandit
    using Off-Policy Replay Evaluation.
    """
    def __init__(self, file_path, variant_col, converted_col):
        self.df = pd.read_csv(file_path)
        self.df.columns = self.df.columns.str.strip()
        self.variant_col = variant_col
        self.converted_col = converted_col
        self.variants = self.df[variant_col].unique().tolist()
        self.total_rows = len(self.df)
        self.current_index = 0

    def has_next(self):
        return self.current_index < self.total_rows

    def get_next_interaction(self):
        row = self.df.iloc[self.current_index]
        self.current_index += 1
        assigned_variant = str(row[self.variant_col])
        conversion = int(row[self.converted_col])
        return assigned_variant, conversion
