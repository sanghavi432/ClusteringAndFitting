import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit
from scipy.stats import t
import numpy as np


def read_clean_transpose_data(file_path):
    """
    Read original data from a CSV file, clean by replacing ".."
    with NaN, and transpose the DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - original_data (pd.DataFrame): Original data read from the CSV file.
    - cleaned_data (pd.DataFrame): Data with ".." replaced by NaN.
    - transposed_data (pd.DataFrame): Transposed DataFrame.
    """
    # Read original data
    original_data = pd.read_csv(file_path)

    # Replace ".." with NaN
    cleaned_data = original_data.replace('..' , pd.NA)

    # Transpose the DataFrame
    transposed_data = cleaned_data.transpose()

    return original_data , cleaned_data , transposed_data


def model_function(x , a , b , c):
    """
        Polynomial model function.

        This function represents a second-degree polynomial model given by:
        y = a * x^2 + b * x + c

        Parameters:
        - x (array-like): Input values.
        - a (float): Coefficient for the quadratic term.
        - b (float): Coefficient for the linear term.
        - c (float): Intercept term.

        Returns:
        - y (array-like): Output values calculated using the polynomial model.
        """
    return a * x**2 + b * x + c


def clusterPlot():
    """
        Generate a scatter plot for K-Means clustering results.

        Parameters:
        - cluster_data_imputed (numpy.ndarray): Imputed and transformed
        data used for clustering.
        - cleaned_data (pd.DataFrame): DataFrame with cluster labels.
        - kmeans (sklearn.cluster.KMeans): Fitted K-Means model.

        Returns:
        None
        """
    # Plotting
    plt.scatter(cluster_data_imputed[: , 1] , cluster_data_imputed[: , 3] ,
                c = cleaned_data["Cluster"] ,
                cmap = 'viridis' , s = 50 , label = 'Data Points')
    plt.scatter(kmeans.cluster_centers_[: , 1] , kmeans.cluster_centers_[: , 3] , c = 'red' ,
                marker = 'X' , s = 200 , label = 'Cluster Centers')
    plt.xlabel("Bank liquid reserves to bank assets ratio")
    plt.ylabel("Binding coverage, all products (%) ")
    plt.title("K-Means Clustering with Imputed Values")
    plt.legend()
    plt.show()


def fittingPlot():
    """
        Generate a scatter plot for polynomial fitting results.

        Parameters:
        - x_data (numpy.ndarray): Input values (e.g., years).
        - y_data (numpy.ndarray): Original data points.
        - future_years (numpy.ndarray): Years for which predictions are made.
        - predicted_values (numpy.ndarray): Predicted values using the fitted
        polynomial model.
        - lower_bound (numpy.ndarray): Lower bound of the 95% confidence interval.
        - upper_bound (numpy.ndarray): Upper bound of the 95% confidence interval.

        Returns:
        None
    """
    # Plotting
    plt.scatter(x_data , y_data , label = 'Original Data')
    plt.plot(future_years , predicted_values , label = 'Fitted Curve' , color = 'red')
    plt.fill_between(x_data , lower_bound , upper_bound , color = 'gray' , alpha = 0.3 ,
                     label = '95% Confidence Interval')
    plt.xlabel("Year")
    plt.ylabel("Forest area (% of land area)")
    plt.title("Polynomial Fit with Confidence Interval")
    plt.legend()
    plt.show()


# Read, clean, and transpose data
file_path = '1448742b-a339-4e45-8cd4-ec5eea86702d_Data.csv'
original_data , cleaned_data , transposed_data = read_clean_transpose_data(file_path)

# Select relevant columns for clustering
cluster_data = cleaned_data[["Forest area (% of land area) [AG.LND.FRST.ZS]" ,
                             "Bank liquid reserves to bank assets ratio (%) [FD.RES.LIQU.AS.ZS]" ,
                             "Binding coverage, all products (%) [TM.TAX.MRCH.BC.ZS]" ,
                             "Births attended by skilled health staff (% of total) [SH.STA.BRTC.ZS]" ,
                             "Bound rate, simple mean, all products (%) [TM.TAX.MRCH.BR.ZS]"]]

# Convert DataFrame to NumPy array, handling 'NAType'
cluster_data_array = cluster_data.apply(pd.to_numeric , errors = 'coerce').values

# Impute missing values with mean
imputer = SimpleImputer(strategy = 'mean')
cluster_data_imputed = imputer.fit_transform(cluster_data_array)

# K-Means clustering
kmeans = KMeans(n_clusters = 7 , random_state = 42)
cleaned_data["Cluster"] = kmeans.fit_predict(cluster_data_imputed)

# Silhouette Score
silhouette_avg = silhouette_score(cluster_data_imputed , cleaned_data["Cluster"])
print(f"Silhouette Score: {silhouette_avg}")

# Replace ".." with NaN
cleaned_data.replace('..' , pd.NA , inplace = True)

# Convert 'NAType' to NaN and then to numeric
numeric_data = cleaned_data.apply(pd.to_numeric , errors = 'coerce')

# Sample data
x_data = np.array(numeric_data['Time'])
y_data = np.array(numeric_data['Forest area (% of land area) [AG.LND.FRST.ZS]'])

# Remove NaN values
mask = ~np.isnan(x_data) & ~np.isnan(y_data)
x_data = x_data[mask]
y_data = y_data[mask]


# Fit the model to the data using curve_fit
params , covariance = curve_fit(model_function , x_data , y_data)

# Calculate the confidence interval
alpha = 0.05  # 95% confidence interval
n = len(y_data)  # number of data points
p = len(params)  # number of parameters

t_value = t.ppf(1 - alpha / 2 , n - p)  # t-value for the confidence interval

param_errs = np.sqrt(np.diag(covariance))
lower_bound = model_function(x_data , *(params - t_value * param_errs))
upper_bound = model_function(x_data , *(params + t_value * param_errs))

# Predict values for the next 10 years
future_years = np.arange(1990 , 2031 , 1)
predicted_values = model_function(future_years , *params)

clusterPlot()

fittingPlot()


