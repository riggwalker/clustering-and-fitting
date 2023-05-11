import pandas as pd
import matplotlib.pyplot as plt

# Load the GDP data into a dataframe
gdp_df = pd.read_csv("API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5454986.csv", skiprows=4)

# Select the GDP data for 2019
gdp_2019 = gdp_df[["Country Name", "2019"]]

# Filter the G5 countries (United States, China, Japan, Germany, United Kingdom)
g5_countries = ["United States", "China", "Japan", "Germany", "United Kingdom"]
g5_gdp = gdp_2019[gdp_2019["Country Name"].isin(g5_countries)]

# Generate a distinctly colored bar plot
colors = ["#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF"]
plt.bar(g5_gdp["Country Name"], g5_gdp["2019"], color=colors)

# Add labels and title to the plot
plt.xlabel("Country")
plt.ylabel("GDP (current US$)")
plt.title("GDP of 5 Big Economies in 2019")

# Rotate the x-axis labels for better visibility
plt.xticks(rotation=45)

# Display the plot
plt.show()


# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Read the dataset
df = pd.read_csv('API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5454986.csv', header=2)

# Selecting the columns to be used for clustering
columns_to_use = [str(year) for year in range(1960, 2021)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Normalize the data
df_norm, df_min, df_max = scaler(df_years[columns_to_use])
df_norm.fillna(0, inplace=True)  # replace NaN values with 0

# Find the optimal number of clusters using the silhouette score
silhouette_scores = []
for i in range(2, 11):  # Considering 2 to 10 clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df_norm)
    labels = kmeans.labels_
    silhouette_scores.append(silhouette_score(df_norm, labels))

# Plot the silhouette scores
plt.plot(range(2, 11), silhouette_scores)
plt.title('Silhouette Score')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.show()


# Using K-means clustering to group data
optimal_clusters = 3  # Modified to 3 clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
df_years['Cluster'] = kmeans.fit_predict(df_norm)

# Add cluster classification as a new column to the dataframe
df_years['Cluster'] = kmeans.labels_

# Plot the clustering results
plt.figure(figsize=(12, 8))
for i in range(optimal_clusters):
    # Select the data for the current cluster
    cluster_data = df_years[df_years['Cluster'] == i]
    # Plot the data
    plt.scatter(cluster_data.index, cluster_data['2019'], label=f'Cluster {i}')

# Plot the cluster centers
cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)
for i in range(optimal_clusters):
    # Plot the center for the current cluster
    plt.scatter(len(df_years), cluster_centers[i, -1], marker='*', s=150, c='black', label=f'Cluster Center {i}')

# Set the title and axis labels
plt.title('GDP equivalent of the current US$ Clustering Results')
plt.xlabel('Country Index')
plt.ylabel('GDP (current US$)')

# Add legend
plt.legend()

# Show the plot
plt.show()


from tabulate import tabulate

# Showing countries in each cluster
for i in range(optimal_clusters):
    cluster_countries = df_years[df_years['Cluster'] == i][['Country Name', 'Country Code']]
    print(f'Countries in Cluster {i}:')
    print(tabulate(cluster_countries, headers='keys', tablefmt='psql'))
    print()


def linear_model(x, a, b):
    return a*x + b
# Define the columns to use
columns_to_use = [str(year) for year in range(1960, 2020)]


# choose a country
country = 'Kenya'

# Extract data for the selected country
country_data = df_years.loc[df_years['Country Name'] == country][columns_to_use].values[0]
x_data = np.array(range(1960, 2020))
y_data = country_data

# Remove any NaN or inf values from y_data
y_data = np.nan_to_num(y_data)

# Fit the linear model
popt, pcov = curve_fit(linear_model, x_data, y_data)

def err_ranges(popt, pcov, x):
    perr = np.sqrt(np.diag(pcov))
    y = linear_model(x, *popt)
    lower = linear_model(x, *(popt - perr))
    upper = linear_model(x, *(popt + perr))
    return y, lower, upper


#showcasing Possible future values and corresponding confidence intervals
x_future = np.array(range(1960, 2041))
y_future, lower_future, upper_future = err_ranges(popt, pcov, x_future)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(x_data, y_data, 'o', label='Data')
plt.plot(x_future, y_future, '-', label='Best Fit')
plt.fill_between(x_future, lower_future, upper_future, color='gray', alpha=0.3, label='Confidence range')
plt.xlabel('Year')
plt.ylabel('GDP (current US$)')
plt.title(f'{country} GDP (current US$) Fitting')
plt.legend()
plt.show()

