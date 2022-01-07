import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler




def create_cluster(df, X, k):
    '''
THIS FUNCTION SCALES THE VARIABLES YOU WANT TO CLUSTER ON AND RETURNS A DF WITH THE CLUSTER PREDICTIONS.
    
INPUTS:
    - df: train df
    - X: df of unscaled features you want to cluster on
    - k: no of clusters

OUTPUTS:
    - df: train df w/cluster column
    - X_scaled: df of scaled features you want to cluster on 
    - scaler: scaler object
    - kmeans: clustering object
    '''
    
    scaler = StandardScaler(copy=True).fit(X)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns.values).set_index([X.index.values])
    
    kmeans = KMeans(n_clusters = k, random_state = 42)
    kmeans.fit(X_scaled)
    
    kmeans.predict(X_scaled)
    df['cluster'] = kmeans.predict(X_scaled)
    df['cluster'] = 'cluster_' + df.cluster.astype(str)
    
    centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X_scaled.columns)
    return df, X_scaled, scaler, kmeans, centroids


def create_scatter_plot(x, y, df, kmeans, X_scaled, scaler, k):
    '''
THIS FUNCTION PLOTS A LABELED SCATTERPLOT OF TWO VARS, WITH COLOR CODED CLUSTERS. 

IF YOU PLOT THE VARS YOU CLUSTERED ON, YOU WILL SEE THE CLUSTER DIVISIONS. 

INPUTS: 
    - x: x var for scatterplot
    - y: y var for scatterplot
    - df: train df w/cluster column
        - df: come back and use this function on validate and test to get cluster columns for modeling!
    - X_scaled: df of scaled features you want to cluster on
    - scaler:: scaler object
    - k: no of clusters
OUTPUT: 
    - viz: labeled scatterplot of x against y, with hue = cluster
    '''
    
    plt.figure(figsize=(18, 10))
    sns.scatterplot(x = x, y = y, data = df, hue = 'cluster')
    
#    THESE LINES OF CODE GIVING ERROR W/X VAR, THEY PLOT THE CENTROIDS | WILL FIX LATER
    #centroids = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns= X_scaled.columns)
    #sns.scatterplot(x, y, ax = plt.gca(), alpha = .30, s = 500, c = 'black')
    #centroids.plot.scatter(x = x, y = y, ax = plt.gca(), alpha = .30, s = 500, c = 'black')
    
    plt.title(f'{x.upper()} and {y.upper()}, Clustered by {X_scaled.columns} | k = {k}')
    plt.show();
    
def X_y_split(train, validate, test, model_features, target):
    '''
    
    '''
    #encode cluster column
    dummy_train_df = pd.get_dummies(train.cluster)
    dummy_validate_df = pd.get_dummies(validate.cluster)
    dummy_test_df = pd.get_dummies(test.cluster)
    
    encoded_train = pd.concat([train , dummy_train_df], axis = 1)
    encoded_validate = pd.concat([validate , dummy_validate_df], axis = 1)
    encoded_test = pd.concat([test , dummy_test_df], axis = 1)
    
    #X and y splits
    X_train = train[model_features]
    y_train = pd.DataFrame(train[target]).reset_index(drop = 'index')
    
    X_validate = validate[model_features]
    y_validate = pd.DataFrame(validate[target]).reset_index(drop = 'index')
    
    X_test = test[model_features]
    y_test = pd.DataFrame(test[target]).reset_index(drop = 'index')
    
    #scale X
    
    
    return encoded_train, encoded_validate, encoded_test, X_train, y_train, X_validate, y_validate, X_test, y_test
    
    
    
    
  
    
    
    
    

