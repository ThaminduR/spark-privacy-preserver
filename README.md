# spark-privacy-preserver

This module provides a simple tool for anonymizing a dataset using PySpark. Given a `spark.sql.dataframe` with relevant metadata mondrian_privacy_preserver generates a anonymized `spark.sql.dataframe`. This provides following privacy preserving techniques for the anonymization.

- K Anonymity
- L Diversity
- T Closeness

Note: Only works with PySpark

## Demo

Jupyter notebook for each of the following modules are included.

- Mondrian Based Anonymity (Single User Anonymization included)
- Clustering Based Anonymity
- Differential Privacy

## Requirements

Requirements for submodules are given under each topic. 

## Installation

### Using pip

Use `pip install spark_privacy_preserver` to install library

### using source code

Clone the repository to your PC and run `pip install .` to build and install the package.

## Basic Mondrian Anoymizing


### Requirements - Basic Mondrian Anonymize

- PySpark 2.4.5. You can easily install it with `pip install pyspark`.
- PyArrow. You can easily install it with `pip install pyarrow`.
- Pandas. You can easily install it with `pip install pandas`.

### K Anonymity

The `spark.sql.dataframe` you get after anonymizing will always contain a extra column `count` which indicates the number of similar rows.
Return type of all the non categorical columns will be string

```python
from spark_privacy_preserver.mondrian_preserver import Preserver #requires pandas

#df - spark.sql.dataframe - original dataframe
#k - int - value of the k
#feature_columns - list - what you want in the output dataframe
#sensitive_column - string - what you need as senstive attribute
#categorical - set -all categorical columns of the original dataframe as a set
#schema - spark.sql.types StructType - schema of the output dataframe you are expecting

df = spark.read.csv(your_csv_file).toDF('age',
    'occupation',
    'race',
    'sex',
    'hours-per-week',
    'income')

categorical = set((
    'occupation',
    'sex',
    'race'
))

feature_columns = ['age', 'occupation']

sensitive_column = 'income'

your_anonymized_dataframe = Preserver.k_anonymize(df,
                                                k,
                                                feature_columns,
                                                sensitive_column,
                                                categorical,
                                                schema)
```

Following code snippet shows how to construct an example schema.
You need to always consider the count column when constructing the schema. Count column is an integer type column.

```python
from spark.sql.type import *

#age, occupation - feature columns
#income - sensitive column

schema = StructType([
    StructField("age", DoubleType()),
    StructField("occupation", StringType()),
    StructField("income", StringType()),
    StructField("count", IntegerType())
])
```

### K Anonymity (without row suppresion)

This function provides a simple way to anonymize a dataset which has an user identification attribute without grouping the rows.    
This function doesn't return a dataframe with the count variable as above function. Instead it returns the same dataframe, k-anonymized. Return type of all the non categorical columns will be string.   
User attribute column **must not** be given as a feature column and its return type will be same as the input type.   
Function takes exact same parameters as the above function. To use this method to anonymize the dataset, instead of calling `k_anonymize`, call `k_anonymize_w_user`.    

### L Diversity

Same as the K Anonymity, the `spark.sql.dataframe` you get after anonymizing will always contain a extra column `count` which indicates the number of similar rows.
Return type of all the non categorical columns will be string

```python
from spark_privacy_preserver.mondrian_preserver import Preserver #requires pandas

#df - spark.sql.dataframe - original dataframe
#k - int - value of the k
#l - int - value of the l
#feature_columns - list - what you want in the output dataframe
#sensitive_column - string - what you need as senstive attribute
#categorical - set -all categorical columns of the original dataframe as a set
#schema - spark.sql.types StructType - schema of the output dataframe you are expecting

df = spark.read.csv(your_csv_file).toDF('age',
    'occupation',
    'race',
    'sex',
    'hours-per-week',
    'income')

categorical = set((
    'occupation',
    'sex',
    'race'
))

feature_columns = ['age', 'occupation']

sensitive_column = 'income'

your_anonymized_dataframe = Preserver.l_diversity(df,
                                                k,
                                                l,
                                                feature_columns,
                                                sensitive_column,
                                                categorical,
                                                schema)
```

### L Diversity (without row suppresion)

This function provides a simple way to anonymize a dataset which has an user identification attribute without grouping the rows.   
This function doesn't return a dataframe with the count variable as above function. Instead it returns the same dataframe, l-diversity anonymized. Return type of all the non categorical columns will be string.    
User attribute column **must not** be given as a feature column and its return type will be same as the input type.   
Function takes exact same parameters as the above function. To use this method to anonymize the dataset, instead of calling `l_diversity`, call `l_diversity_w_user`.  

### T - Closeness

Same as the K Anonymity, the `spark.sql.dataframe` you get after anonymizing will always contain a extra column `count` which indicates the number of similar rows.
Return type of all the non categorical columns will be string

```python
from spark_privacy_preserver.mondrian_preserver import Preserver #requires pandas

#df - spark.sql.dataframe - original dataframe
#k - int - value of the k
#l - int - value of the l
#feature_columns - list - what you want in the output dataframe
#sensitive_column - string - what you need as senstive attribute
#categorical - set -all categorical columns of the original dataframe as a set
#schema - spark.sql.types StructType - schema of the output dataframe you are expecting

df = spark.read.csv(your_csv_file).toDF('age',
    'occupation',
    'race',
    'sex',
    'hours-per-week',
    'income')

categorical = set((
    'occupation',
    'sex',
    'race'
))

feature_columns = ['age', 'occupation']

sensitive_column = 'income'

your_anonymized_dataframe = Preserver.t_closeness(df,
                                                k,
                                                t,
                                                feature_columns,
                                                sensitive_column,
                                                categorical,
                                                schema)

```

### T Closeness (without row suppresion)

This function provides a simple way to anonymize a dataset which has an user identification attribute without grouping the rows.  
This function doesn't return a dataframe with the count variable as above function. Instead it returns the same dataframe, t-closeness anonymized. Return type of all the non categorical columns will be string.   
User attribute column **must not** be given as a feature column and its return type will be same as the input type.   
Function takes exact same parameters as the above function. To use this method to anonymize the dataset, instead of calling `t_closeness`, call `t_closeness_w_user`.  

### Single User K Anonymity

This function provides a simple way to anonymize a given user in a dataset. Even though this doesn't use the mondrian algorithm, function is included in the `mondrian_preserver`. User identification attribute and the column name of the user identification atribute is needed as parameters.   
This doesn't return a dataframe with count variable. Instead this returns the same dataframe, anonymized for the given user. Return type of user column and all the non categorical columns will be string.

```python
from spark_privacy_preserver.mondrian_preserver import Preserver #requires pandas

#df - spark.sql.dataframe - original dataframe
#k - int - value of the k
#user - name, id, number of the user. Unique user identification attribute.
#usercolumn_name - name of the column containing unique user identification attribute.
#sensitive_column - string - what you need as senstive attribute
#categorical - set -all categorical columns of the original dataframe as a set
#schema - spark.sql.types StructType - schema of the output dataframe you are expecting
#random - a flag by default set to false. In a case where algorithm can't find similar rows for given user, if this is set to true, slgorithm will randomly select rows from dataframe.

df = spark.read.csv(your_csv_file).toDF('name',
    'age',
    'occupation',
    'race',
    'sex',
    'hours-per-week',
    'income')

categorical = set((
    'occupation',
    'sex',
    'race'
))

sensitive_column = 'income'

user = 'Jon'

usercolumn_name = 'name'

random = True

your_anonymized_dataframe = Preserver.anonymize_user(df,
                                                k,
                                                user,
                                                usercolumn_name,
                                                sensitive_column,
                                                categorical,
                                                schema,
                                                random)

```
## Introduction to Differential Privacy

Differential privacy is one of the data preservation paradigms similar to K-Anonymity, T-Closeness and L-Diversity.
It alters each value of actual data in a dataset according to specific constraints set by the owner and produces a 
differentially-private dataset. This anonymized dataset is then released for public utilization.

**ε-differential privacy** is one of the methods in differential privacy. Laplace based ε-differential privacy is 
applied in this library. The method states that the randomization should be according the **epsilon** (ε) 
(should be >0) value set by data owner. After randomization a typical noise is added to the dataset. It is calibrated 
according to the **sensitivity** value (λ) set by the data owner. 

Other than above parameters, a third parameter **delta** (δ) is added into the mix to increase accuracy of the 
algorithm. A **scale** is computed from the above three parameters and a new value is computed.

> scale = λ / (ε - _log_(1 - δ))

> random_number = _random_generator_(0, 1) - 0.5

> sign = _get_sign_(random_number)

> new_value = value - scale × sign × _log_(1 - 2 × _mod_(random_number))

In essence above steps mean that a laplace transform is applied to the value and a new value is randomly selected 
according to the parameters. When the scale becomes larger, the deviation from original value will increase.

## Achieving Differential Privacy
    
### Requirements

* Python

Python versions above Python 3.6 and below Python 3.8 are recommended. The module is developed and tested on: 
Python 3.7.7 and pip 20.0.2. (It is better to avoid Python 3.8 as it has some compatibility issues with Spark)

* Apache Spark

Spark 2.4.5 is recommended. 

* Java

Java 8 is recommended. Newer versions of java are incompatible with Spark.

The module is developed and tested on:
    
    java version "1.8.0_231"
    Java(TM) SE Runtime Environment (build 1.8.0_231-b11)
    Java HotSpot(TM) 64-Bit Server VM (build 25.231-b11, mixed mode)

Make sure the following Python packages are installed:
1. PySpark: ```pip install pyspark==2.4.5```
2. PyArrow: ```pip install pyarrow==0.17.1```
3. IBM Differential Privacy Library: ```pip install diffprivlib==0.2.1```
4. MyPy:  ```pip install mypy==0.770```

### Procedure

Step by step procedure on how to use the module with explanation is given in the following 
notebook: **differential_privacy_demo.ipynb**

1. Create a Spark Session. Make sure to enable PyArrow configuration.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master('local') \
    .appName('differential_privacy') \
    .config('spark.some.config.option', 'some-value') \
    .getOrCreate()

spark.conf.set('spark.sql.execution.arrow.enabled', 'true')
```

2. Create a Spark DataFrame (sdf). 

Generate an sdf with random values. It is better to manually specify the **schema** of sdf so as to avoid 
any *TypeErrors*.

Here I will generate an sdf with 3 columns: *'Numeric'*, *'Rounded_Numeric'*, *'Boolean'* and 10,000 rows 
to show 3 ways of using DPLib.

```python
from random import random, randint, choice
from pyspark.sql.types import *

# generate a row with random numbers of range(0, 100000) and random strings of either 'yes' or 'no'
def generate_rand_tuple():
    number_1 = randint(0, 100000) + random()
    number_2 = randint(0, 100000) + random()
    string = choice(['yes', 'no'])
    return number_1, number_2, string

data = [generate_rand_tuple() for _ in range(100000)]

schema = StructType([
    StructField('Number', FloatType()),
    StructField('Rounded_Number', DoubleType()),
    StructField('Boolean', StringType())
])

sdf = spark.createDataFrame(data=data, schema=schema)
sdf.show(n=5)
```

3. Setup and configure **DPLib**

DPLib can work with numbers and binary strings. To anonymize a number based column, you have to setup the column 
category as *'numeric'*. To anonymize a string based column, you have to setup the column category as *'boolean'*.

3.1 Initializing the module

The module takes in 3 optional parameters when initializing: *Spark DataFrame*, *epsilon* and *delta*. Module can also 
be initialized without any parameters and they can be added later.

```python
from spark_privacy_preserver.differential_privacy import DPLib

epsilon = 0.00001
delta = 0.5
sensitivity = 10

# method 1
dp = DPLib(global_epsilon=epsilon, global_delta=delta, sdf=sdf)
dp.set_global_sensitivity(sensitivity=sensitivity)

# method 2
dp = DPLib()
dp.set_sdf(sdf=sdf)
dp.set_global_epsilon_delta(epsilon=epsilon, delta=delta)
dp.set_global_sensitivity(sensitivity=sensitivity)

```

**Note:** The reason behind the word *global* in above functions

Suppose the user want to anonymize 3 columns of a DataFrame with same epsilon, delta and sensitivity and another 
column with different parameters. Now all the user has to do is to set up global parameters for 3 columns and 
local parameters for 4th column. 

This will simplify when multiple columns of a DataFrame have to be processed with same parameters.

3.2 Configuring columns

User can configure columns with column specific parameters. Column specific parameters will be given higher priority 
over global parameters if explicitly specified.

parameters that can be applied to method *set_column()*:
1. column_name: name of column as string -> compulsory
2. category: category of column. can be either *'numeric'* or *'boolean'* -> compulsory
3. epsilon: column specific value -> optional
4. delta: column specific value -> optional
5. sensitivity: column specific value -> optional
6. lower_bound: set minimum number a column can have. can only be applied to category *'numeric'* -> optional
7. upper_bound: set maximum number a column can have. can only be applied to category *'numeric'* -> optional
8. label1: string label for a column. can only be applied to category *'binary'* -> optional
9. label2: string label for a column. can only be applied to category *'binary'* -> optional
10. round: value by which to round the result. can only be applied to category *'numeric'* -> optional

```python
dp.set_column(column_name='Number', 
              category='numeric')
# epsilon, delta, sensitivity will be taken from global parameters and applied.

dp.set_column(column_name='Rounded_Number', 
              category='numeric',
              epsilon=epsilon * 10,
              sensitivity=sensitivity * 10,
              lower_bound=round(sdf.agg({'Rounded_Number': 'min'}).collect()[0][0]) + 10000,
              upper_bound=round(sdf.agg({'Rounded_Number': 'max'}).collect()[0][0]) - 10000,
              round=2)
# epsilon, sensitivity will be taken from user input instead of global parameters
# delta will be taken from global parameters.

dp.set_column(column_name='Boolean',
              category='boolean',
              label1='yes',
              label2='no',
              delta=delta if 0 < delta <= 1 else 0.5)
# sensitivity will be taken from user input instead of global parameters
# epsilon will be taken from global parameters.
# 'boolean' category does not require delta

dp._DPLib__columns
```

3.3 Executing

```python
# gets first 20 rows of DataFrame before anonymizing and after anonymizing
sdf.show()

dp.execute()

dp.sdf.show()
```

As you can see, there is a clear difference between original DataFrame and anonymized DataFrame.

1. Column *'Number'* is anonymized but the values are not bound to a certain range. The algorithm produces the result 
with maximum precision as it can achieve.

2. Column *'Rounded_Number'* is both anonymized and bounded to the values set by user. As you can see, the values 
never rise above upper bound and never become lower than lower bound. Also they are rounded to 2nd decimal place as set.

3. Column *'Boolean'* undergoes through a mechanism that randomly decides to flip to the other binary value or not, 
in order to satisfy differential privacy.