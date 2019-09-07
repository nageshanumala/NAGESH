"""
Last amended: 01/02/2019

Objective: Simple experiments using pandas groupby


By “group by” we are referring to a process involving one or
more of the following steps:
    i)   Splitting the data into groups based on some criteria.
    ii)  Applying a function to each group independently.
    iii) Combining the results into a data structure.

Out of these, the split step is the most straightforward.
In fact, in many situations we may wish to split the data
set into groups and do something with those groups. In the
apply step, we might wish to one of the following:

    i)   Aggregation: compute a summary statistic (or statistics) for each group.
    ii)  Transformation: perform some group-specific computations and return a like-indexed object.
    iii) Filtration: discard some groups, according to a group-wise computation that evaluates True or False.



"""

# 1.0 Call libraries
import pandas as pd
import numpy as np

# 2.0 Define a simple dataframe
df = pd.DataFrame([('bird', 'Falconiformes', 389.0, 21.2),
                  ('bird', 'Psittaciformes', 24.0, 23.5),
                  ('mammal', 'Carnivora', 80.2, 29.0),
                  ('mammal', 'Primates', np.nan, 30.6),
                  ('mammal', 'Carnivora', 58, 40.8),
                  ('fish', 'Whale', 89, 120.8),
                  ('fish', 'Shark', 78, 80.8)],
                  index=['falcon', 'parrot', 'lion', 'monkey', 'leopard','whale','shark'],
                  columns=('class', 'order', 'max_speed', 'max_wt'))

df

#############
## 3. Splitting
#############
# Various ways to groupby
# default grouping is by is axis=0
# Collectively we refer to the grouping objects as the keys.
grouped = df.groupby('class')      # Same as: df.groupby(['class'])
grouped1 = df.groupby(['class', 'order'])

grouped      # <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x7f2f944e2128>
grouped1     # <pandas.core.groupby.groupby.DataFrameGroupBy object at 0x7f2f944e27b8>



###########################
##4. GroupBy object attributes
###########################
grouped.groups           # Describes groups dictionary
grouped2.groups          # Dict object
len(grouped)             # How many items are there in the group



# 4.1 Iterating through the groups
#     Peeping into each basket
for name, group in grouped:
    print(name)
    print(group)

# 4.2 Out of these multiple boxes/groups
#     A single group can be selected using get_group():
grouped.get_group('fish')


##############
## 5. Aggregating
#############
# Once the GroupBy object has been created,
# several methods are available to perform
# a computation on the grouped data.

# 5.1
grouped['max_speed'].sum()     # keys are sorted

"""
# Summary functions are
mean() 	   Compute mean of groups
sum() 	   Compute sum of group values
size() 	   Compute group sizes
count()    Compute count of group
std() 	   Standard deviation of groups
var() 	   Compute variance of groups
sem() 	   Standard error of the mean of groups
describe() Generates descriptive statistics
first()    Compute first of group values
last() 	   Compute last of group values
nth() 	   Take nth value, or a subset if n is a list
min() 	   Compute min of group values
max()      Compute max of group values
"""

# 5,2 With grouped Series you can also pass a
# list or dict of functions to do
# aggregation with, outputting a DataFram

grouped['max_speed'].agg([np.sum, np.mean, np.std])

# 5.3 By passing a dict to aggregate you can apply a
#     different aggregation to the columns of a DataFrame:

grouped.agg({'max_speed': np.sum,
             'max_wt': np.std })

##########################################################
