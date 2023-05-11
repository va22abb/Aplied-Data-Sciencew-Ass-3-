# -*- coding: utf-8 -*-
"""
Created on Sat May 6 15:34:18 2023

@author: Victoria
"""

# import libraries
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit as cf


# load data into pandas data frame
def read_data(fname):
    data = pd.read_csv(fname)
    df = pd.DataFrame(data)
    dataT = pd.read_csv(fname, header=None, index_col=0).T
    dfT = pd.DataFrame(dataT)
    dfT = dfT.rename(columns={"Country Name": "Year"})
    return df, dfT


# The function convert def to numbers
def convert_to_numbers(df, column_list):
    df[column_list] = df[column_list].apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    return df


def objective(x, a, b, c):
    return (a * x) + (b * x ** 2) + c


def curve_fit_plot(dft, df1t, country):
    country_new = "New " + country
    dft = convert_to_numbers(dft, [country])
    df1t = convert_to_numbers(df1t, [country])
    popt, _ = cf(objective, dft[country], df1t[country])
    df1t[country_new] = objective(dft[country], *popt)
    plt.plot(dft[country], df1t[country], color='red')
    plt.plot(dft[country], df1t[country_new], color='blue')
    cr = np.std(df1t[country]) / np.sqrt(len(df1t[country]))
    plt.fill_between(dft[country], (df1t[country] - cr), (df1t[country] + cr),
                     color='b', alpha=0.1)
    plt.xlabel(dft.columns[0])
    plt.ylabel(df1t.columns[0])
    plt.legend(["Expected", "Predicted"], loc="upper right")
    plt.title("Curve Fit of " + country)
    
def kmeans_cluster(df_fit, no_clusters, labels):
    kmeans = KMeans(n_clusters=no_clusters, random_state=0)
    labels = kmeans.fit_predict(df_fit)
    cen = kmeans.cluster_centers_
    plt.scatter(df_fit.iloc[:, 0], df_fit.iloc[:, 1], c=labels, cmap="Accent")
    for ic in range(no_clusters):
        xc, yc = cen[ic, :]
        plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title("K-means Clustering")


# Input file paths
fname1 = "Life expectancy at birth, total.csv"
fname2 = "Mortality caused by road traffic injury.csv"
# Importing warnings so that it may ignore warnings
warnings.filterwarnings('ignore')

# Invoking the function to get the data
educationDF, dft = read_data(fname2)
govexpenditureDF, df1t = read_data(fname1)
labels = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]

# Taking the required years for plotting into an array
years = [
    "1990",
    "2000",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018"]

# Invoking convert_to_numbers methods to change the data to numeric form
df = convert_to_numbers(educationDF, years)
df1 = convert_to_numbers(govexpenditureDF, years)


