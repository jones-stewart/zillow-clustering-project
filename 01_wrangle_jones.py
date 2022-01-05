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
        query = '''
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
        url = 'mysql+pymysql://{user}:{password}@{host}/"zillow"'
        zillow = pd.read_sql(query, url)
        zillow.to_csv('zillow.csv')
        return zillow

##########################################################################################| WRANGLE FUNCTION

##########################################################################################| PREPARE FUNCTION