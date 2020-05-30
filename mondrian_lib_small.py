from pyspark.sql import SparkSession
from pyspark.sql.types import *
from mondrian_preserver import Preserver


spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

data = [[6, '1', 'test1', 'x', 20],
        [6, '1', 'test1', 'y', 30],
        [8, '2', 'test2', 'x', 50],
        [8, '2', 'test3', 'x', 45],
        [8, '1', 'test2', 'y', 35],
        [4, '2', 'test3', 'y', 20]]

cSchema = StructType([StructField("column1", IntegerType()),
                      StructField("column2", StringType()),
                      StructField("column3", StringType()),
                      StructField("column4", StringType()),
                      StructField("column5", IntegerType())])
df = spark.createDataFrame(data, schema=cSchema)


# variables
categorical = set((
    'column2',
    'column3',
    'column4'
))
sensitive_column = 'column4'
feature_columns = ['column2', 'column3', 'column5']
schema = StructType([
    StructField("column1", IntegerType()),
    StructField("column2", StringType()),
    StructField("column3", StringType()),
    StructField("column4", StringType()),
    StructField("column5", StringType()),
])
user = 6
usercolumn_name = "column1"
k = 2

# anonymizing
dfn = Preserver.k_anonymize_w_user(
    df, k, feature_columns, sensitive_column, categorical, schema)
dfn.show()
spark.stop()
