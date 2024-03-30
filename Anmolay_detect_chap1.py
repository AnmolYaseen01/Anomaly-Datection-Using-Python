import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import zscore, median_abs_deviation
from pyod.models.mad import MAD

df=pd.read_csv("your_dataframe.csv")

# Print 5-number summary
print(df.describe())

# Find the square root of the length of df
n_bins = int(np.sqrt(len(df)))

# Create a histogram
plt.figure(figsize=(8, 4))
plt.hist(df, bins=n_bins)
plt.show()

# Create a list of consecutive integers
integers = range(len(df))

# Plot a scatterplot
plt.figure(figsize=(16, 8))
plt.scatter(integers, df, c='red', alpha=0.5)
plt.show()

# Create a boxplot of df
plt.boxplot(df, whis=3.5)
plt.show()

# Calculate the 25th and 75th percentiles
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)

# Find the IQR
IQR = q3 - q1
factor = 2.5

# Calculate the lower limit
lower_limit = q1 - (IQR * factor)

# Calculate the upper limit
upper_limit = q3 + (IQR * factor)

# Create a mask for values lower than lower_limit
is_lower = df < lower_limit

# Create a mask for values higher than upper_limit
is_higher = df > upper_limit

# Combine the masks to filter for outliers
outliers = df[is_higher | is_lower]

# Count and print the number of outliers
print(len(outliers))

# Find the z-scores of df
scores = zscore(df)

# Check if the absolute values of scores are over 3
is_over_3 = np.abs(scores) > 3

# Use the mask to subset df
outliers = df[is_over_3]

print(len(outliers))

# Find the median absolute deviation (MAD) score
mad_score = median_abs_deviation(df)

# Initialize MAD with a threshold of 3.5
mad = MAD(threshold=3.5)

# Reshape df to make it 2D 
df_reshaped = df.values.reshape(-1, 1)

# Fit and predict outlier labels on df_reshaped
labels = mad.fit_predict(df_reshaped)

# Filter for outliers
outliers = df[labels == 1]

print(len(outliers))