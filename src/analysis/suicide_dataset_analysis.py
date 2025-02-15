import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("../../data/00_raw/gpt2_raw_encoded_suicide_detection.csv")  # Update with your actual file path
df["token_count"] = df["encoded_text"].apply(lambda x: len(x))

# outlier threshold 95th percentile
upper_limit = np.percentile(df["token_count"], 95)
filtered_counts = df[df["token_count"] <= upper_limit]["token_count"]

plt.figure(figsize=(10, 6))
plt.hist(filtered_counts, bins=50, color="blue", edgecolor="black", alpha=0.7)
plt.xlabel("Token Count (Outliers Removed)")
plt.ylabel("Frequency")
plt.title("Token Count Distribution (Outliers ≤ 95th Percentile)")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plot_path = "../../plots/suicide_token_count_hist.png"
plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Histogram saved at: {plot_path}")