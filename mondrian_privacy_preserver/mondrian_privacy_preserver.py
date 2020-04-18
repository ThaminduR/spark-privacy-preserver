import pandas as pd
from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from utils.utility import *

# class KAnonymity:

#     """
#     @PARAMS
#     df - pandas dataframe
#     partition - parition for whic to calculate the spans
#     scale: if given, the spans of each column will be divided
#             by the scale for that column
#     """

#     def __get_spans(self, df, categorical, partition, scale=None):

#         columns = list(df.columns)
#         spans = {}
#         for column in df.columns:
#             if column in categorical:
#                 span = len(df[column][partition].unique())
#             else:
#                 span = df[column][partition].max()-df[column][partition].min()
#             if scale is not None:
#                 span = span/scale[column]
#             spans[column] = span
#         return spans

#     """
#     @PARAMS
#     df - pandas dataframe
#     partition - parition for whic to calculate the spans
#     column: column to split
#     """

#     def __split(self, df, categorical, partition, column):
#         dfp = df[column][partition]
#         if column in categorical:
#             values = dfp.unique()
#             lv = set(values[:len(values)//2])
#             rv = set(values[len(values)//2:])
#             return dfp.index[dfp.isin(lv)], dfp.index[dfp.isin(rv)]
#         else:
#             median = dfp.median()
#             dfl = dfp.index[dfp < median]
#             dfr = dfp.index[dfp >= median]
#             return (dfl, dfr)

#     def __is_k_anonymous(self, partition, k):

#         if len(partition) < k:
#             return False
#         return True

#     """
#     @PARAMS
#     df - pandas dataframe
#     feature_column - list of column names along which to partitions the dataset
#     scale - column spans
#     is_valid - function to check the validity of a partition
#     """

#     def __partition_dataset(self, df, k, categorical, feature_columns, sensitive_column, scale, is_valid):

#         finished_partitions = []
#         partitions = [df.index]
#         while partitions:
#             partition = partitions.pop(0)
#             spans = self.__get_spans(df[feature_columns],
#                                      categorical, partition, scale)
#             for column, span in sorted(spans.items(), key=lambda x: -x[1]):
#                 lp, rp = self.__split(df, categorical, partition, column)
#                 if not is_valid(lp, k) or not is_valid(rp, k):
#                     continue
#                 partitions.extend((lp, rp))
#                 break
#             else:
#                 finished_partitions.append(partition)
#         return finished_partitions

#     def __agg_categorical_column(self, series):
#         return ','.join(set(series))

#     def __agg_numerical_column(self, series):
#         return series.mean()

#     @classmethod
#     def anonymizer(cls, df, k, feature_columns, sensitive_column, categorical, max_partitions=None):
#         aggregations = {}

#         for name in df.columns:
#             if name not in categorical:
#                 df[name] = pd.to_numeric(df[name])

#         full_spans = cls.__get_spans(df, categorical, df.index)
#         partitions = cls.__partition_dataset(
#             df, k, categorical, feature_columns, sensitive_column, full_spans, __is_k_anonymous)
#         for column in feature_columns:
#             if column in categorical:
#                 aggregations[column] = cls.__agg_categorical_column
#             else:
#                 aggregations[column] = cls.__agg_numerical_column
#         rows = []

#         for i, partition in enumerate(partitions):
#             if i % 100 == 1:
#                 print("Finished {} partitions.".format(i))
#             if max_partitions is not None and i > max_partitions:
#                 break
#             grouped_columns = df.loc[partition].assign(
#                 m=1).groupby('m').agg(aggregations, squeeze=False)
#             sensitive_counts = df.loc[partition].groupby(
#                 sensitive_column).agg({sensitive_column: 'count'})
#             values = grouped_columns.iloc[0].to_dict()
#             for sensitive_value, count in sensitive_counts[sensitive_column].items():
#                 if count == 0:
#                     continue
#                 values.update({
#                     sensitive_column: sensitive_value,
#                     'count': count,

#                 })
#                 rows.append(values.copy())
#         dfn = pd.DataFrame(rows)
#         pdfn = dfn.sort_values(feature_columns+[sensitive_column])
#         return pdfn


def K_anonymizer(df, k, feature_columns, sensitive_column, categorical, max_partitions=None):
    aggregations = {}

    for name in df.columns:
        if name not in categorical:
            df[name] = pd.to_numeric(df[name])

    full_spans = get_spans(df, categorical, df.index)
    partitions = partition_dataset(
        df, k, categorical, feature_columns, sensitive_column, full_spans, is_k_anonymous)
    for column in feature_columns:
        if column in categorical:
            aggregations[column] = agg_categorical_column
        else:
            aggregations[column] = agg_numerical_column
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


class Preserver:

    @staticmethod
    def k_anonymize(pdf, k, feature_columns, sensitive_column, categorical, schema):
        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            df = K_anonymizer(pdf, k, feature_columns,
                                 sensitive_column, categorical)
            return df

        #new_df = pdf.withColumn("common", lit(0))
        return pdf.groupby('age').apply(anonymize)
