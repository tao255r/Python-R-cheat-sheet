"""
General Notes:
    * Using = is not creating a copy, it`s pointing 2 different names to the same location
"""

"""
Get and Change Working Directory
"""
import os
os.getcwd() # get current working directory
os.chdir('C:/xxx') # change currecnt working directory

"""
Reading BIG csv file line by line
"""
def show_line(filename, linenumber):
    file = open(filename, 'r')
    counter = 0
    for line in file:
        if counter == linenumber:
            print line
        if counter > linenumber:
            break
        counter += 1
    file.close()

"""
Pandas Data Frame & Data Manipulation
"""
import pandas as pd

# Create Data Frame from dictionary
data1 = {'ID': [1001, 1002, 1003],
         'Name': ['John', 'James', 'Ken'],
         'Age': [30, 28, 36]}
df = pd.DataFrame(data1, columns = ['ID', 'Name', 'Age']) # add columns to make sure the order of columns arranged

# Create Data Frame from list
data1 = [['John', 1001, 30],
         ['Tim', 1002, 40],
         ['Allen', 1003, 25]]

df = pd.DataFrame(data1, columns = ['Name', 'ID', 'Age'])

# show all columns's column types
df.dtypes

# list out all column names
list(df)

# Select data frame by columns
target = df[['col1', 'col2']]
target = df.ix[:, ('col1', 'col2')]
target = df.iloc[0:10]

# Select data frame by rows & columns
df.ix[(df['col1'] == 100) & (df['col2'] == 101) | (df['col3'] == 102), 'colname']

# Select unique value of rows
target = df.loc[~df.duplicated()]
# Check unique value of a column
df['colName'].unique()
# Check unique value of multiple columns
pd.unique(df[['clusters', 'names']].values.ravel())

# Create new column based on existing column
df['names'] = np.where(df.clusters == 0, 'A',
                       np.where(df_clusters.clusters == 1, 'B',
                               np.where(df_clusters.clusters == 2, 'C', 'D')))

# Dummy variables
## append dummy variables to the origional dataframe
pd.get_dummies(df, prefix='Category_', columns=['col1', 'col2'])
## append dummy variables to the origional dataframe, remove the first level
pd.get_dummies(df, prefix='Category_', columns=['col1', 'col2'], drop_first = True)

# Merge / Concat / Join
## More about this: https://pandas.pydata.org/pandas-docs/stable/merging.html
## Merge
    # how : {‘left’, ‘right’, ‘outer’, ‘inner’}, default ‘inner’
new_df = pd.merge(df1, df2, right_index=True, left_index=True) # Merge based on indexes
new_df = pd.merge(df1, df2, on = 'StoreID', how = 'left') # merge based on 1 column
new_df = pd.merge(df1, df2, on = ['StoreID', 'LocationID'], how = 'inner') # merge based on n columns
new_df = pd.merge(df1, df2, left_on = ['StoreID', 'LocationID'], 
                            right_on = ['StoreNumber', 'LocationNumber'], how = 'inner') # merge on different columns
new_df = pd.merge(df1, df2, on='col', how='left', suffixes=('_left', '_right')) # Merge while adding a suffix

## Concat 
    # axis : {0/’index’, 1/’columns’}, default 0
d1 = {'name': ['Tom','Ken','Tim'],
      'age' : [32,26,53]}
d2 = {'name': ['A','K','T'],
      'age' : [3,6,3]}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)

pd.concat([df1,df2], axis = 1)
    #   age name    age name
    #0  32  Tom 3   A
    #1  26  Ken 6   K
    #2  53  Tim 3   T
pd.concat([df1,df2], axis = 0)
    #   age name
    #0  32  Tom
    #1  26  Ken
    #2  53  Tim
    #0  3   A
    #1  6   K
    #2  3   T

# Group by in Pandas
df_new = df.groupby('col1').count()
df_new = df.groupby(['col1','col2']).sum().reset_index() # reset_index() help clean up the leves created by groupby

# Pivot table vs melt
## Pivot
df_new = df.pivot_table(index=['propcode', 'propbeds', 'ratetype','wknum'], 
                        columns='day', values='cumnormdemandpct').reset_index()

    # We try to turn something like this:
        client  propcode    propfp  pctile  adj
        test    127306      1B1B-A1  0      100
        test    127306      1B1B-A1  0.1    99.11
        test    127306      1B1B-A1  0.2    98.32
        test    127306      1B1B-A1  0.3    97.05
        test    127306      1B1B-A1  0.4    95.53
        test    127306      1B1B-A1  0.5    93.99
        test    127306      1B1B-A1  0.6    92.16
        test    127306      1B1B-A1  0.7    89.9
        test    127306      1B1B-A1  0.8    85.5
        test    127306      1B1B-A1  0.9    65.05
        test    127306      1B1B-A1  1      0
        test    127306      2B1B-A4  0      100
        test    127306      2B1B-A4  0.1    98.91
        test    127306      2B1B-A4  0.2    97.82
        test    127306      2B1B-A4  0.3    96.73
        test    127306      2B1B-A4  0.4    95.64
        test    127306      2B1B-A4  0.5    94.55
        test    127306      2B1B-A4  0.6    93.46
        test    127306      2B1B-A4  0.7    90.4
        test    127306      2B1B-A4  0.8    88.65333333
        test    127306      2B1B-A4  0.9    69.23
        test    127306      2B1B-A4  1      0

    # to something like this:
        client  propcode    propfp  0   0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1
        venterra    127306  1B1B-A123*  100 99.11   98.32   97.05   95.53   93.99   92.16   89.9    85.5    65.05   0
        venterra    127306  2B1B-A4 100 98.91   97.82   96.73   95.64   94.55   93.46   90.4    88.65333333 69.23   0

# Python code for it would be:
df_fix = df.pivot_table(index = ['client','propcode','propfp'], columns='pctile', values='adj')
df_fix.columns = df_fix.columns.get_level_values('pctile')
df_fix.reset_index(inplace=True)

# From pivot table format back to origional format could use melt
v = ['client','propcode','propfp']
df = pd.melt(df_fix, id_vars = v, var_name = 'pctile', value_name = 'adj')


"""
h2o.ai in python
"""
import h2o
# Start the h2o clusters / shut down clusters
h2o.init()
h2o.cluster().shutdown()
h2o.cluster().show_status() # check cluster status

# Data exchange between pandas and h2o
df_h2o = h2o.H2OFrame(df) # import pandas dataframe to h2o dataframe
df = df_h2o.as_data_frame() # export h2o dataframe to pandas dataframe

# Data mining algorithms 
## K-mean clustering
from h2o.estimators.kmeans import H2OKMeansEstimator ## import the estimator
model = H2OKMeansEstimator(k=10, init="Random", seed=2, standardize=True) # describe the model
model.train(x=df_h2o.col_names, training_frame = df_h2o) # train the model (scikt-learn use model.fit, h2o use model.train)
predicted = model.predict(df_h2o) # prediction using built model

## Random Forest
from h2o.estimators.random_forest import H2ORandomForestEstimator
model = H2ORandomForestEstimator(ntrees = n, max_depth = d)
    # Train and predict similar to K-Mean

"""
PostgreSQL
"""
import psycopg2
import pandas as pd

# Build connection to postgresql database
conn = psycopg2.connect("dbname='database name' user='user name' host='host name' password='password'")

# Querying and fetch data to a pandas dataframe
df = pd.read_sql_query('select * from table',con=conn)

# Export pandas data to PSQL table
    from sqlalchemy import create_engine
    engine = create_engine('postgresql+psycopg2://user_name:password@host_name/database_name')
    df.to_sql('table_name_in_db', engine, if_exists='append', index = False)

"""
Python statistic
"""
# One Way ANOVA (f statistic)
import scipy.stats as stats
stats.f_oneway(group1, group2, group3, ...)

# Adding new column with calculation or condition on existing column 
    # method 1:
    newColumn = []
    for index, row in df.iterrows():
        cvalue = 'yes' if df['a'] == 0 else 'no'
        newColumn.append(cvalue)
    df['new_column_name'] = pd.Series(np.array(newColumn), index=df.index) 

    # method 2 using np:
    import numpy as np
    df['new_column_name'] = np.where((df['a'] == 0), 'yes', 'no')

# calculate quartiles
    # Assume values is a serious of ints
    values = map(int, raw_input().strip().split(' '))
    values.sort()

    def median(value, n):
        if n%2 == 0:
            m = (value[n/2-1] + value[n/2])/2 # will round down to closest int
            return m
        else:
            m = value[n/2] # will round down to closest int
            return m

    if N%2 != 0:
        low = values[:N/2]
        up = values[N/2+1:]
    else:
        low = values[:N/2]
        up = values[N/2:]
        
    q1 = median(low, len(low))
    q2 = median(values, N)
    q3 = median(up, len(up))

"""
Others
"""
# Python scripts call main function
    if __name__ == "__main__":
        fun()

# Jupyter magic
%%time
%matplotlib inline


# Python data frame to R data frame through feather
# The Feather API is designed to make reading and writing data frames as easy as possible. 
    #In R, the code might look like:
    library(feather)
    path <- "my_data.feather"
    write_feather(df, path)
    df <- read_feather(path)

    #Analogously, in Python, we have:
    import feather
    path = 'my_data.feather'
    feather.write_dataframe(df, path)
    df = feather.read_dataframe(path)
