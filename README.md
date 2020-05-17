# spark-privacy-preserver

This module provides a simple tool for anonymizing a dataset using Spark. Given a `spark.sql.dataframe` with relevant metadata mondrian_privacy_preserver generates a anonymized `spark.sql.dataframe`. This provides following privacy preserving techniques for the anonymization.

- K Anonymity
- L Diversity
- T Closeness

## Demo

A Jupyter notebook can be found here

`TODO: add jupyter notebook`

## Requirements

- Apache Spark (versions 2.4.5 and higher are supported).
- PyArrow. You can easily install it with `pip install pyarrow`.
- Pandas. You can easily install it with `pip install pandas`.

## Installation

## Usage - Basic Mondrian

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

This function provides a simple way to anonymize a dataset which has a user identification attribute without grouping the rows.    
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

This function provides a simple way to anonymize a dataset which has a user identification attribute without grouping the rows.   
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

This function provides a simple way to anonymize a dataset which has a user identification attribute without grouping the rows.  
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
