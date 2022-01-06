##########################################################################################| EXPLORE.PY

##########################################################################################| IMPORTS
import wrangle_jones

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

##########################################################################################| UNIVARIATE EXPLORE
def visualize_outliers():
    '''
THIS FUNCTION PLOTS TWO ROWS OF BOXPLOTS, ALLOWING US TO VISUALIZE THE DATA 
BEFORE AND AFTER OUTLIERS REMOVED.
    '''
    
    outlier_cols = ['baths', 'beds', 'sqft', 'fullbaths', 'tax_value', 'logerror']
    raw_df = wrangle_jones.acquire_zillow()
    clean_df = wrangle_jones.wrangle_zillow(wrangle_jones.acquire_zillow())
    
    plt.figure(figsize = (32, 4))
    
    #top row of boxplots will show data with outliers, each column will be a variable
    for i, col in enumerate(outlier_cols):
        
        plot_number = i + 1
        plt.subplot(1, len(outlier_cols), plot_number)
        plt.title(f'{col} with Outliers')
        sns.boxplot(data = raw_df, y = raw_df[col])
        plt.grid(False);
        
    plt.figure(figsize = (32, 4))
    
    #bottom row will show data with outliers removed
    for i, col in enumerate(outlier_cols):
        
        plot_number = i + 1
        plt.subplot(1, len(outlier_cols), plot_number)
        plt.title(f'{col} with Outliers Removed')
        sns.boxplot(data = clean_df, y = clean_df[col])
        plt.grid(False);
        

def var_distributions(df):
    '''
THIS FUNCTION PLOTS HISTOGRAMS FOR EACH VARIABLE IN THE DATAFRAME
THAT IS FED INTO THE FUNCTION, ALLOWING US TO SEE THE SHAPE OF THE
DATA'S DISTRIBUTION. 
    '''
    
    df.hist(figsize = (18, 9))
    plt.show()
    
##########################################################################################| Clustering
