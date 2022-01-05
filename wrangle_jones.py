##########################################################################################| WRANGLE.PY

##########################################################################################| IMPORTS
import os

from env import host, user, password

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

##########################################################################################| ACQUIRE FUNCTION
def acquire_zillow():
    '''
THIS FUNCTION CHECKS FOR A LOCAL ZILLOW CSV FILE AND WRITES IT TO A DF. 
IF A LOCAL ZILLOW CSV FILE DOES NOT EXIST IT RUNS THE QUERY TO ACQUIRE
THE DATA USING THE SQL SERVERS CREDENTIALS FROM THE ENV FILE THROUGH
THE CONNECTION STRING.
    '''
    
    if os.path.isfile('zillow.csv'):
        zillow = pd.read_csv('zillow.csv', index_col = 0)
        return zillow
    
    else: 
        db = 'zillow'
        query  = '''
            SELECT bathroomcnt as baths, bedroomcnt as beds, calculatedfinishedsquarefeet as sqft, fips, fullbathcnt as fullbaths, latitude,
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
        url = f'mysql+pymysql://{user}:{password}@{host}/{db}'
        zillow = pd.read_sql(query, url)
        zillow.to_csv('zillow.csv')
        return zillow

##########################################################################################| WRANGLE FUNCTIONS
def wrangle_zillow(df):
    '''
THIS FUNCTION TAKES IN A RAW DATAFRAME AND PERFORMS THE FOLLOWING DATA CLEANING:
    1) FILTER OUT NON-SINGLE UNIT OBSERVATIONS
    2) DROPS NULLS
    3) ADDS COLUMNS WITH LABELED COUNTIES BASED ON FIPS CODES
    4) CORRECTING DTYPES
    5) CREATE AGE COLUMN
    6) DROP COLUMNS
    '''
    
    
    #filter single units by properylandusetype, bath, bed, and sqft count, and unit count
    df = df[df.propertylandusetypeid.isin([261, 262, 263, 264, 266, 268, 273, 276, 279])]
    df = df[(df.baths > 0) & (df.beds > 0) & (df.sqft > 300)]
    df = df[df.unitcnt == 1]
    
    #dropping null rows and columns with > 50% of values missing
#    df = df.dropna(axis = 1, thresh = .5 * len(df))
#    df = df.dropna(thresh = .5 * len(df.columns))
    df.dropna(inplace = True)
    
    #label fips counties
    df['fips'] = df.fips.astype(int)
    df['fips_loc'] = df.fips.replace({6037:'Los Angeles, CA', 6059:'Orange, CA', 6111:'Ventura, CA'})
    
    #correcting dtypes
    df[['beds', 'sqft', 'fullbaths', 'latitude', 'longitude', 'rooms', 'yearbuilt', 'unitcnt']] = df[['beds', 'sqft',\
                                                                                                    'fullbaths', 'latitude', 'longitude', 'rooms', 'yearbuilt', 'unitcnt']].astype('int')
    
    #create age column from yearbuilt
    df['age'] = 2022 - df.yearbuilt
    
    #drop columns 
    df = df.drop(columns = ['propertylandusetypeid', 'fips', 'transactiondate', 'yearbuilt'])
    
    
    return df
    

##########################################################################################| PREPARE FUNCTION
