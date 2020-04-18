from pyspark.sql import SparkSession
from pyspark.sql.types import *
from mondrian_privacy_preserver import Preserver
logFile = "file:///C:/spark-2.4.5-bin-hadoop2.7/bin/pythonProject/spark-privacy-preserver/data/adult.all.txt"
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

#reading csv
df = spark.read.csv(logFile).toDF('age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
    'native-country',
    'income')

#variables
categorical = set((
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'sex',
    'native-country',
    'race',
    'income',
))
feature_columns = ['age', 'occupation']
sensitive_column = 'income'
schema = StructType([
    StructField("age", DoubleType()),
    StructField("occupation", StringType()),
    StructField("income", StringType()),
    StructField("count", IntegerType())
])

#anonymizing
dfn = Preserver.k_anonymize(df,3,feature_columns, sensitive_column, categorical, schema)
dfn.head(20)
dfn.show()

spark.stop()
