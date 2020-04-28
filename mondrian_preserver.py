
from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from utils.utility import *


def k_anonymizer(df, k, feature_columns, sensitive_column, categorical):

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


def l_diversity(df, k, l, feature_columns, sensitive_column, categorical):

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, l, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


def t_closenes(df, k, t, feature_columns, sensitive_column, categorical):
    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None, t,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


class Preserver:

    @staticmethod
    def k_anonymize(df, k, feature_columns, sensitive_column, categorical, schema):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            a_df = k_anonymizer(pdf, k, feature_columns,
                                sensitive_column, categorical)
            return a_df

        return df.groupby().apply(anonymize)

    @staticmethod
    def l_diversity(df, k, l, feature_columns, sensitive_column, categorical, schema):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            a_df = l_diversity(pdf, k, l, feature_columns,
                               sensitive_column, categorical)
            return a_df

        # TODO compare perfomances

        # new_df = pdf.withColumn("_common888column_", lit(0))
        return df.groupby().apply(anonymize)
