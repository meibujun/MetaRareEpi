import sys
import csv
import numpy as np
from pathlib import Path

np.random.seed(42)
n = 10000

asym = np.exp(-np.random.exponential(1.5, n))
spa = np.exp(-np.random.exponential(1.1, n))

asym = np.clip(asym, 1e-300, 1)
spa = np.clip(spa, 1e-300, 1)

Path("results").mkdir(exist_ok=True)
with open("results/fedcspa_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["pvalue_asymptotic", "pvalue_fedcspa"])
    for a, s in zip(asym, spa):
        writer.writerow([a, s])
print("Fake results created as CSV.")
