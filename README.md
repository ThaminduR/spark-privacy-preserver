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

## Usage

### Basic Mondrian K Anonymity

The `spark.sql.dataframe` you get after anonymizing will always contain a extra column `count` which indicates the number of similar rows. 

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
You need to always consider the count column when constructing the schema. Count column is a integer type column.

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

