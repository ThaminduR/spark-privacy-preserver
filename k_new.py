import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from pyspark.sql.types import *

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.enabled", "true")
logFile = "file:///C:/spark-2.4.5-bin-hadoop2.7/bin\spark-privacy-preserver/adult.all.txt"


def __get_spans(df, categorical, partition, scale=None):

    columns = list(df.columns)
    spans = {}
    for column in df.columns:
        if column in categorical:
            span = len(df[column][partition].unique())
        else:
            span = df[column][partition].max()-df[column][partition].min()
        if scale is not None:
            span = span/scale[column]
        spans[column] = span
    return spans


def __split(df, categorical, partition, column):
    dfp = df[column][partition]
    if column in categorical:
        values = dfp.unique()
        lv = set(values[:len(values)//2])
        rv = set(values[len(values)//2:])
        return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
    else:
        median = dfp.median()
        dfl = dfp.index[dfp < median]
        dfr = dfp.index[dfp >= median]
        return (dfl, dfr)


def __is_k_anonymous(partition, k):

    if len(partition) < k:
        return False
    return True


def __partition_dataset(df, k, categorical, feature_columns, sensitive_column, scale, is_valid):

    finished_partitions = []
    partitions = [df.index]
    while partitions:
        partition = partitions.pop(0)
        spans = __get_spans(df[feature_columns], categorical, partition, scale)
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            lp, rp = __split(df, categorical, partition, column)
            if not is_valid(lp, k) or not is_valid(rp, k):
                continue
            partitions.extend((lp, rp))
            break
        else:
            finished_partitions.append(partition)
    return finished_partitions


def __agg_categorical_column(series):
    return ','.join(set(series))


def __agg_numerical_column(series):
    return series.mean()


def k_anonymize(df, k, feature_columns, sensitive_column, _categorical, max_partitions=None):
    aggregations = {}

    for name in df.columns:
        if name not in _categorical:
            df[name] = pd.to_numeric(df[name])

    full_spans = __get_spans(df, _categorical, df.index)
    partitions = __partition_dataset(
        df, k, _categorical, feature_columns, sensitive_column, full_spans, __is_k_anonymous)
    categorical = _categorical
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = __agg_categorical_column
        else:
            aggregations[column] = __agg_numerical_column
    rows = []

    for i, partition in enumerate(partitions):
        if i % 100 == 1:
            print("Finished {} partitions.".format(i))
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df.loc[partition].assign(
            m=1).groupby('m').agg(aggregations, squeeze=False)
        sensitive_counts = df.loc[partition].groupby(
            sensitive_column).agg({sensitive_column: 'count'})
        values = grouped_columns.iloc[0].to_dict()
        for sensitive_value, count in sensitive_counts[sensitive_column].items():
            if count == 0:
                continue
            values.update({
                sensitive_column: sensitive_value,
                'count': count,

            })
            rows.append(values.copy())
    dfn = pd.DataFrame(rows)
    pdfn = dfn.sort_values(feature_columns+[sensitive_column])
    return pdfn
    

    
df = spark.read.csv(logFile).toDF('age','workclass','fnlwgt','education',
                                'education-num','marital-status','occupation',
                                'relationship','race','sex','capital-gain',
                                'capital-loss','hours-per-week','native-country','income')

#User defined variables
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
k = 3
feature_columns = ['age', 'occupation']
sensitive_column = 'income'

#schema for pandas_udf
schema = StructType([
    StructField("age", DoubleType()),
    StructField("occupation", StringType()),
    StructField("income", StringType()),
    StructField("count", IntegerType())
])

#added for groupby  
# new_df = df.withColumn("common", lit(0))


@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def anonymize(pdf):
    df =  k_anonymize(pdf, k, feature_columns, sensitive_column, categorical)
    return df

df.groupby('age').apply(anonymize).show()

