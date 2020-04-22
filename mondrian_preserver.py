
from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from utils.utility import *


def k_anonymizer(df, k, feature_columns, sensitive_column, categorical):

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, None,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


def l_diversity(df, k, l, feature_columns, sensitive_column, categorical):

    full_spans = get_full_span(df, categorical)
    partitions = partition_dataset(
        df, k, l,  categorical, feature_columns, sensitive_column, full_spans)

    return anonymizer(df, partitions, feature_columns, sensitive_column, categorical)


class Preserver:

    @staticmethod
    def k_anonymize(pdf, k, feature_columns, sensitive_column, categorical, schema):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            df = k_anonymizer(pdf, k, feature_columns,
                              sensitive_column, categorical)
            return df

        # TODO compare perfomances
        new_df = pdf.withColumn("_common888column_", lit(0))
        return new_df.groupby('_common888column_').apply(anonymize)

        # return pdf.groupby('_common888column_').apply(anonymize)

    @staticmethod
    def l_diversity(pdf, k, l, feature_columns, sensitive_column, categorical, schema):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymize(pdf):
            df = l_diversity(pdf, k, l, feature_columns,
                             sensitive_column, categorical)
            return df
        # TODO compare perfomances
        new_df = pdf.withColumn("_common888column_", lit(0))
        return new_df.groupby('_common888column_').apply(anonymize)

        # return pdf.groupby('_common888column_').apply(anonymize)
