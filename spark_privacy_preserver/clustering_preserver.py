from pyspark.sql.types import *
from pyspark.sql.functions import PandasUDFType, lit, pandas_udf
from .clustering_anonymizer import Kanonymizer, LDiversityAnonymizer, TClosenessAnonymizer

class Preserver:
    @staticmethod
    def k_anonymize(df, schema, QI, SA, CI, k, mode='', center_type='fbcg', return_mode='Not_equal', iter=1):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):
            anonymizer = Kanonymizer(pdf, QI, SA, CI)
            a_df = anonymizer.anonymize(
                k=k, mode=mode, center_type=center_type, return_mode=return_mode, iter=iter)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def l_diverse(df, schema, quasi_identifiers, sensitive_attributes, write_to_file=False, l=2):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):

            anonymizer = LDiversityAnonymizer(
                pdf, quasi_identifiers, sensitive_attributes, write_to_file)
            a_df = anonymizer.anonymize(l=l)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def t_closer(df, schema, quasi_identifiers, sensitive_attributes, t=0.3, write_to_file=False, verbose=1):

        @pandas_udf(schema, PandasUDFType.GROUPED_MAP)
        def anonymizer(pdf):
            anonymizer = TClosenessAnonymizer(
                pdf, quasi_identifiers, sensitive_attributes, write_to_file)
            a_df = anonymizer.anonymize(t=t)
            return a_df

        return df.groupby().apply(anonymizer)

    @staticmethod
    def test(df,QI, SA, CI, k,mode='', center_type='fbcg', return_mode='Not_equal', iter=1):
        anonymizer = Kanonymizer(df,QI,SA,CI)
        df = anonymizer.anonymize(k=k, mode=mode, center_type=center_type, return_mode=return_mode, iter=iter)
        return df