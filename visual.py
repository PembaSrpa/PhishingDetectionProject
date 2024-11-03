import pandas as pd  # Data manipulation library
import numpy as np  # Library for numerical operations
import matplotlib.pyplot as plt  # Plotting library
import seaborn as sns  # Statistical data visualization library
import warnings  # To manage warnings
import os  # For file and directory operations

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Create a directory for saving diagrams if it doesn't exist
os.makedirs('diagrams', exist_ok=True)

# Load dataset
d = pd.read_csv("dataset.csv")

# Summary:
# This code performs exploratory data analysis (EDA) on a dataset containing URLs.
# It visualizes the relationship between various features and the status of URLs (Phishing or Legitimate).
# Diagrams include scatter plots, count plots, a correlation heatmap, distribution plots, box plots, and a pie chart.

# Exploratory Data Analysis

# Scatter plots to visualize relationships between numeric features and the status
plt.figure(figsize=(5, 5))
plt.subplot(221)
sns.scatterplot(x=d["web_traffic"], y=d["status"])
plt.xlabel("Web Traffic")
plt.ylabel("Status")
plt.title("Web Traffic vs Status")
plt.savefig("diagrams/web_traffic_vs_status.png")

plt.subplot(222)
sns.scatterplot(x=d["domain_age"], y=d["status"])
plt.xlabel("Domain Age")
plt.ylabel("Status")
plt.title("Domain Age vs Status")
plt.savefig("diagrams/domain_age_vs_status.png")

plt.subplot(223)
sns.scatterplot(x=d["length_url"], y=d["status"])
plt.xlabel("URL Length")
plt.ylabel("Status")
plt.title("URL Length vs Status")
plt.savefig("diagrams/url_length_vs_status.png")

plt.subplot(224)
sns.scatterplot(x=d["nb_hyphens"], y=d["status"])
plt.xlabel("Number of Hyphens")
plt.ylabel("Status")
plt.title("Number of Hyphens vs Status")
plt.savefig("diagrams/number_of_hyphens_vs_status.png")
plt.show()

# Count plots for categorical features against status
fig, ax = plt.subplots(3, 3, sharey=True, figsize=(15, 8))
plt.subplots_adjust(hspace=0.3)

# Create count plots for different features
sns.countplot(data=d, x='nb_com', hue='status', ax=ax[0, 0])
ax[0, 0].set_title("Number of Comments vs Status")
sns.countplot(data=d, x='prefix_suffix', hue='status', ax=ax[0, 1])
ax[0, 1].set_title("Prefix/Suffix vs Status")
sns.countplot(data=d, x='whois_registered_domain', hue='status', ax=ax[0, 2])
ax[0, 2].set_title("WHOIS Registered Domain vs Status")
sns.countplot(data=d, x='nb_www', hue='status', ax=ax[1, 0])
ax[1, 0].set_title("Number of WWW vs Status")
sns.countplot(data=d, x='nb_dots', hue='status', ax=ax[1, 1])
ax[1, 1].set_title("Number of Dots vs Status")
sns.countplot(data=d, x='phish_hints', hue='status', ax=ax[1, 2])
ax[1, 2].set_title("Phishing Hints vs Status")
sns.countplot(data=d, x='ip', hue='status', ax=ax[2, 0])
ax[2, 0].set_title("IP vs Status")
sns.countplot(data=d, x='nb_slash', hue='status', ax=ax[2, 1])
ax[2, 1].set_title("Number of Slashes vs Status")
sns.countplot(data=d, x='shortening_service', hue='status', ax=ax[2, 2])
ax[2, 2].set_title("Shortening Service vs Status")

# Save count plots
plt.savefig("diagrams/count_plots.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))

# Select only numeric columns for correlation
numeric_columns = d.select_dtypes(include=[np.number])  # Keep only numeric columns
correlation_matrix = numeric_columns.corr()  # Compute the correlation matrix

# Plot the correlation heatmap
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.savefig("diagrams/correlation_heatmap.png")
plt.show()

# Distribution plots for numerical features
numerical_features = ['web_traffic', 'domain_age', 'length_url', 'nb_hyphens']  # Add your numerical features here
plt.figure(figsize=(15, 10))

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)  # Create a 2x2 grid for the plots
    sns.histplot(d[feature], kde=True)  # Histogram with KDE
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig("diagrams/distribution_plots.png")
plt.show()

# Box plots for numerical features against status
plt.figure(figsize=(15, 10))

for i, feature in enumerate(numerical_features, 1):
    plt.subplot(2, 2, i)  # Create a 2x2 grid for the plots
    sns.boxplot(x='status', y=feature, data=d)  # Box plot for each numerical feature
    plt.title(f'Box Plot of {feature} by Status')
    plt.xlabel('Status')
    plt.ylabel(feature)

plt.tight_layout()
plt.savefig("diagrams/box_plots.png")
plt.show()

# Pie chart for phishing vs legitimate distribution
status_counts = d['status'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(status_counts, labels=['Legitimate', 'Phishing'], autopct='%1.1f%%', startangle=90, colors=['#66c2a5', '#fc8d62'])
plt.title("Distribution of Phishing vs Legitimate URLs")
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
plt.savefig("diagrams/pie_chart_distribution.png")
plt.show()

# Prepare data for feature importances
x = d.iloc[:, :-1]  # Features
y = d['status']  # Target variable
x.drop(['url'], axis=1, inplace=True)  # Drop URL column if exists

# Display the final message for the user
print("Exploratory Data Analysis completed. Diagrams saved in 'diagrams' directory.")
