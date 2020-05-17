from pyspark.sql import SparkSession
from pyspark.sql.types import *
from mondrian_preserver import Preserver

logFile = "file:///C:/spark-2.4.5-bin-hadoop2.7/bin/pythonProject/spark-privacy-preserver/data/calls.csv"
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# reading csv
df = spark.read.csv(logFile,header= True)

# variables
categorical = set((
    'user', 
    'other', 
    'direction', 
    'timestamp'
))
feature_columns = ['user', 'other','direction']
sensitive_column = 'timestamp'

schema = StructType([
    StructField("user", StringType()),
    StructField("other", StringType()),
    StructField("direction", StringType()),
    StructField("duration", DoubleType()),
    StructField("timestamp", StringType()),
    # StructField("count", IntegerType())
])

# anonymizing
# dfn = Preserver.k_anonymize(df, 3, feature_columns,
#                             sensitive_column, categorical, schema)
# dfn.show(50)

user = '07641036117'
usercolumn_name = "user"
k = 3

# anonymizing
dfn = Preserver.anonymize_user(df, k, user, usercolumn_name, sensitive_column, categorical,schema)
dfn.show()

spark.stop()
