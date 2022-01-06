# FUNCTIONS FOR WRANGLING ZILLOW DATA

import pandas as pd

def get_url(db):
    '''
    This function takes in a database name and returns a url (using the specified 
    database name as well as host, user, and password from env.py) for use in the 
    pandas.read_sql() function.
    '''
    from env import host, user, password
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def acquire_zillow():
    '''
    This function pulls data from the SQL zillow database and caches that data to a csv
    for later data retrieval. It takes no arguments and returns a dataframe of zillow data.
    '''
    import os
    if os.path.isfile('zillow.csv'):
        zillow = pd.read_csv('zillow.csv', index_col=0)
        return zillow
    else:        
        sql = '''
            SELECT bathroomcnt as baths, bedroomcnt as beds, calculatedfinishedsquarefeet as sq_ft, fips, fullbathcnt as fullbaths, latitude,
                   longitude, roomcnt as rooms, yearbuilt, taxvaluedollarcnt as tax_value, garagecarcnt, logerror, transactiondate, 
                   unitcnt, propertylandusetypeid
            FROM properties_2017
            LEFT JOIN predictions_2017 pred USING(parcelid)
            LEFT JOIN airconditioningtype USING(airconditioningtypeid)
            LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
            LEFT JOIN buildingclasstype USING(buildingclasstypeid)
            LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
            LEFT JOIN propertylandusetype USING(propertylandusetypeid)
            LEFT JOIN storytype USING(storytypeid)
            LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
            WHERE latitude IS NOT NULL
            AND longitude IS NOT NULL
            AND transactiondate LIKE "2017%%"
            AND pred.id IN (SELECT MAX(id)
            FROM predictions_2017
            GROUP BY parcelid
            HAVING MAX(transactiondate));
            '''
        zillow = pd.read_sql(sql, get_url('zillow'))
        zillow.to_csv('zillow.csv')
        return zillow

def cols_missing_rows(df):
    '''
    This function takes in a dataframe and returns a dataframe of column names, the number
    of rows that column is missing, and the percentage of rows that column is missing.
    '''
    df = pd.DataFrame(data={'num_rows_missing':df.isnull().sum(), 
              'pct_rows_missing':df.isnull().sum()/len(df)}, index=df.columns)
    return df

def rows_missing_cols(df):
    '''
    This function takes in a dataframe and returns a dataframe of the number of columns
    missing from a row, the percentage of columns missing from a row, and the number of
    rows that are missing that number/percentage of columns.
    '''
    df = pd.DataFrame({'num_cols_missing':df.isnull().sum(axis=1).value_counts().index,
                       'pct_cols_missing':df.isnull().sum(axis=1).value_counts().index/len(df.columns),
                       'num_rows':df.isnull().sum(axis=1).value_counts()}).reset_index(drop=True)
    return df

def only_single_units(zillow):
    '''
    This function takes in the zillow dataframe and removes any properties not believed
    to be single-unit properties. It returns zillow without those properties.
    '''
    zillow_filt = zillow[zillow.propertylandusetypeid.isin([261, 262, 263, 264, 266, 268, 273, 276, 279])]
    zillow_filt = zillow_filt[(zillow.baths > 0) & (zillow.sq_ft > 300) & (zillow.beds > 0)]
    zillow_filt = zillow_filt[(zillow_filt.unitcnt == 1) | (zillow_filt.unitcnt.isnull())]
    return zillow_filt

def handle_missing_values(df, prop_req_col, prop_req_row):
    '''
    This function takes in a dataframe, a max proportion of null values for each 
    column, and a max proportion of null values for each row. It returns the 
    dataframe less any rows or columns with more than the max proportion of nulls.
    '''
    df = df.dropna(axis=1, thresh=prop_req_col*len(df))
    df = df.dropna(thresh=prop_req_row*len(df.columns))
    return df

def label_fips(zillow):
    zillow['fips'] = zillow.fips.astype(int)
    zillow['fips_loc'] = zillow.fips.replace({6037:'Los Angeles, CA',
                       6059:'Orange, CA',
                       6111:'Ventura, CA'})
    return zillow

def remove_outliers(df, cols, k):
    '''
    This function takes in a list of column names from a dataframe and a 
    k-value which is used to specify the upper and lower bounds for
    removing outliers. It returns the dataframe with the outliers removed.
    '''
    # make for loop to remove outliers in each column
    for col in cols:
        # get quartiles
        q1, q3 = df[col].quantile([.25, .75])
        # compute iqr
        iqr = q3 - q1
        # get cutoff points for removing outliers
        upper = q3 + k * iqr
        lower = q1 - k * iqr
        # remove outliers
        df = df[(df[col]>lower)&(df[col]<upper)]
    return df

def split_data(df):
    '''
    This function takes in a dataframe and splits it into three dataframes.
    It returns these dataframes in this order: train, validate, test.
    Train makes up 56% of the total observations, validate 24%, and test 20%.
    '''
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, random_state=123)
    train, validate = train_test_split(train, test_size=0.3, random_state=123)
    return train, validate, test

def standard_scale_data(train, validate, test, scaled_cols):
    '''
    This function takes in train, validate, and test dataframes and a list of columns 
    to be scaled. It returns those dataframes with standard-scaled data columns added.
    '''
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(train[scaled_cols])
    train[[f'{col}_scaled' for col in scaled_cols]] = scaler.transform(train[scaled_cols])
    validate[[f'{col}_scaled' for col in scaled_cols]] = scaler.transform(validate[scaled_cols])
    test[[f'{col}_scaled' for col in scaled_cols]] = scaler.transform(test[scaled_cols])
    return train, validate, test

def wrangle_zillow(prop_req_col, prop_req_row):
    '''
    This function wrangles zillow data. It takes in thresholds for null values which are
    used to drop columns and rows with too many nulls. The function returns a dataframe.
    '''
    zillow = handle_missing_values(only_single_units(acquire_zillow()), prop_req_col, prop_req_row)
    zillow = zillow.drop(columns=['unitcnt', 'propertylandusetypeid']).dropna()
    zillow = label_fips(zillow)
    from datetime import date
    zillow['age'] = date.today().year - zillow.yearbuilt
    return zillow