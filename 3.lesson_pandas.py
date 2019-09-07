"""
Last amended: 1st February, 2019
Myfolder:  /home/ashok/Documents/1.basic_lessons


Reference: http://pandas.pydata.org/pandas-docs/stable/dsintro.html#series
           https://docs.python.org/2/tutorial/introduction.html#unicode-strings


Objectives:
	i)  Data structures in pandas: Series, DataFrame and Index
	ii) Data structures usage


"""

import pandas as pd
import numpy as np
import os


########Series#############
## A. Creating Series
# 10. Series is one-dimensional (columnar) index labeled
#      array. Labels are referred as index. Series may have
#        dtype as float64, int64 or object

# 10.1 Exercises
s = pd.Series([2,4,8,10,55])
s
type(s)

# 10.2 Exercise
ss=[23,45,56]
h=pd.Series(ss)
h

# 10.3 OR generate it as:
h=pd.Series(list(range(23,30,2)))

## B. Simple Operations
# 10.4 Exercise
s+h
s*h
s-h

(s+h)[1]       # Note the indexing starts from 0
s*h[2]

s.mean()
s.std()
s.median()

## C. Series as ndarray
 # Also series behaves as ndarray
 # Series acts very similarly to a ndarray, and is a valid argument to most NumPy functions.
np.mean(s)
np.median(s)
np.std(s)


## D. Indexing in series
# 10.5 Exercise
d=pd.Series([4,5], index=['a','b'])
e=pd.Series([6,7], index=['f','g'])
f=pd.Series([9,10], index=['a','b'])
d+e  # All NaN
d+f


# Add d and e above as:
d.reset_index(drop = True, inplace = True)   # drop = False, adds existing index as
                                             # a new column and makes it a DataFrame
d
e.reset_index(drop = True, inplace = True)
d + e


## E. Accessing Series
# 10.6 Exercise
j= pd.Series(np.random.normal(size=7))

k=j[j>0]
k=j[j>np.mean(j)]

# 10.7 Exercise
k = pd.Series(np.random.normal(size=7),index=['a','b','c','d','e','f','a'])
k['a']
k[:2]   # Show first two or upto 2nd index (0 and 1)
k[2:]   # Start from 2nd index
k[2:4]  # Start from IInd index upto 4th index
k[2:4].mean()


# 10.7.1  SURPRISE HERE!
k = pd.Series(np.random.normal(size=7),index=[0,2,5,3,4,1,6])
k[0]                 # Access by index-name
k[1]                 # Access by index-name
k[:2]                # Access by position
k[[0,1,2]]           # Access by index-name
k.take([0,1,2])      # Access by position


# 10.8 Exercise
# A series is like a dictionary. Can be accessed by its index (key)
e=pd.Series(np.random.uniform(0,5,7), index=['a','b','c','d','e','f','g'])
e
e['a' : 'e']
k['a' : 'd']   # All values from 'a' to 'd'
k['b' : 'd']
e+k


## F. Forget this part
# 10.9 Exercise
# Generating an index automatically
t=pd.Series(list(range(15)), index = [chr(i) for i in range(97,97+15)]
t


# Exercise 7:
j=pd.Series(np.random.uniform(1,7,7) )
j+e


######## DataFrame ###########

'''
DataFrame is a 2-dimensional labeled data structure with columns
of potentially different types. You can think of it like a spreadsheet
or SQL table, or a dict of Series objects. It is generally the most
commonly used pandas object. Like Series, DataFrame accepts many
different kinds of input.
'''

## G. Create Dataframe
# Exercise 9:
# Creating dataframes
a=pd.Series(np.random.normal(size=9))
b=pd.Series(np.random.uniform(size=9))
c=pd.DataFrame ({'one' : a, 'two' : b})
d=pd.Series({'one' : a, 'two' : b})
e=pd.Series({'one' : 1, 'two' : 2 })

a
b
c
type(c)
# Create a column
c['three'] = 'bar'
c
d
e

d['one']
d['two']
d['one'][1]
d['one'][:3]
d['one'] + d['two']

data=pd.DataFrame( {'one' : pd.Series({'a' : 3, 'b' : 4 }), 'two' : pd.Series({'a' : 34, 'b' : 67})} )
data


'''
No of rows in dataframe
    Dataframe dim
    Dataframe columns
    Dataframe head
    df tail
    df summary
    df structure
    add column to df
    access/slice df
    filter df
'''

df=pd.DataFrame({'one' : pd.Series(np.random.uniform(size=100)),
                 'two' : pd.Series(np.random.uniform(size=100)),
                 'three' : pd.Series(np.random.uniform(size=100)),
                 'four' : pd.Series(np.random.uniform(size=100)),
                 'five' : pd.Series([chr(i) for i in range(94,194)])   })
df
df.shape
df.columns
df.values
df.head()
df.tail()
df.dtypes		# Data structure
df.info()
df.describe()
df.describe(include = ['O'])


# Selection by Labels:
df[15:23]         	# Slice rows from inex 15 upto 22 element
df.loc[:,['one','three']]
df.loc[2:4,['one','three']]
df.loc[2:4,'four':'two'].head()

'''
Selection By Position:
pandas provides a suite of methods in order to get purely
integer based indexing. The semantics follow closely python
and numpy slicing. These are 0-based indexing. When slicing,
the start bounds is included, while the upper bound is excluded.
Trying to use a non-integer, even a valid label will raise a IndexError.

The .iloc attribute is the primary access method. The following are valid inputs:
	An integer e.g. 5
	A list or array of integers [4, 3, 0]
	A slice object with ints 1:7
	A boolean array
'''

df.iloc[3:5, 1:2]  # 3:5 implies start from 3rd pos uptil 5-1=4th pos
df.iloc[3:5, 1:3]  # Display column numbers 2nd and 3rd
df.iloc[3:5]		# Display all columns
df.iloc[3:5, :]		# Display all columns
df.iloc[1,1]        # Same as df[1,1:2]. Treat 1 as lower bound
df.iloc[[3,5,7],[1,3]]		# Specific rows and columns
df1=df[df.three > 0.8].head()   # Boolean indexing
df1=df[df.three > 0.8 ].head()   # Boolean indexing
df1=df[(df.three > 0.8) & (df.one > 4)]
df1=df[(df.three > 0.8) | (df.one > 4)]




os.getcwd()
os.chdir ("/home/ashokharnal/Documents/data_analysis/python_in_class/")
surveys_df=pd.read_csv("surveys.csv")
type(surveys_df)
surveys_df.dtypes
surveys_df.head()
surveys_df.tail()
surveys_df.shape
type(surveys_df.shape)
surveys_df.columns
surveys_df.columns.values
pd.unique(surveys_df['month'])	# Unique values
pd.unique(surveys_df.species_id)
surveys_df.species_id.describe()
# The groupby command is powerful in that
#  it allows us to quickly generate summary stats.
