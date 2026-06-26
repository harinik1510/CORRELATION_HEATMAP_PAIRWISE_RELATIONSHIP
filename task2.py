import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\CLLG STUFF\PYTHON INTERNSHIP\DATASCIENCE\student_data.csv")

print("Dataset Preview:")
print(df.head())


numeric_df = df.select_dtypes(include=np.number)
corr_matrix = numeric_df.corr(method='pearson')

print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(10, 8))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

sns.pairplot(numeric_df)
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()


corr_pairs = corr_matrix.unstack()

corr_pairs = corr_pairs[corr_pairs != 1]
corr_pairs = corr_pairs.sort_values()

strongest_negative = corr_pairs.head(1)
strongest_positive = corr_pairs.tail(1)

print("\nStrongest Negative Correlation:")
print(strongest_negative)

print("\nStrongest Positive Correlation:")
print(strongest_positive)