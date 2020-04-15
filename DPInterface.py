from numbers import Real
import pandas as pd

from diffprivlib.mechanisms import LaplaceTruncated
from diffprivlib.mechanisms import Binary

import swifter


class DPInterface:

    def __init__(self, global_epsilon=None, global_delta=0.0, df=None):

        self.df = df
        self.__columns = {}

        self.__epsilon = None
        self.__delta = None

        self.__sensitivity = None

        if global_epsilon is not None and self.__check_epsilon_delta(global_epsilon, global_delta):
            self.__epsilon = float(global_epsilon)
            self.__delta = float(global_delta)

    @staticmethod
    def __check_epsilon_delta(epsilon, delta):
        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise TypeError("Epsilon and delta must be numeric")

        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= delta <= 1:
            raise ValueError("Delta must be in range [0, 1]")

        if epsilon == 0 and delta == 0:
            raise ValueError("Epsilon and Delta cannot both be zero")

        return True

    @staticmethod
    def __check_sensitivity(sensitivity):
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        return True

    @staticmethod
    def __check_labels(df, column_name, label1, label2):
        if not isinstance(label1, str) or not isinstance(label2, str):
            raise TypeError("Labels must be strings.")

        if len(label1) == 0 or len(label2) == 0:
            raise ValueError("Labels must be non-empty strings")

        if label1 == label2:
            raise ValueError("Labels must not match")

        labels = df[column_name].unique()
        if len(labels) is 2 and label1 in labels and label2 in labels:
            return True
        else:
            raise ValueError("Column has multiple unique labels")

    def set_global_epsilon_delta(self, epsilon, delta=0.0):
        if self.__check_epsilon_delta(epsilon, delta):
            self.__epsilon = float(epsilon)
            self.__delta = float(delta)

    def set_global_sensitivity(self, sensitivity):
        if self.__check_sensitivity(sensitivity):
            self.__sensitivity = float(sensitivity)

    @staticmethod
    def __check_bounds(lower, upper):

        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        return True

    def set_df(self, df):
        self.df = df

    def add_column(self, column_name, category,
                   epsilon=None, delta=None,
                   sensitivity=None, lower_bound=None, upper_bound=None, round=None,
                   label1=None, label2=None):

        if self.df is None:
            raise ValueError("Add an eligible DataFrame before adding columns")

        if column_name not in list(self.df.columns):
            raise ValueError("Cannot find column in given DataFrame")

        if category not in ['numeric', 'boolean']:
            raise ValueError("Cannot find category in available list")

        column = {'category': category}

        if epsilon is None:
            if self.__epsilon is not None:
                epsilon = self.__epsilon
            else:
                raise ValueError("Epsilon must be set")

        if delta is None:
            if self.__delta is not None:
                delta = self.__delta
            else:
                delta = 0.0

        if epsilon is not None and delta is not None and self.__check_epsilon_delta(epsilon, delta):
            column['epsilon'] = epsilon
            column['delta'] = delta

        if category is 'numeric':

            if sensitivity is None: sensitivity = self.__sensitivity
            if self.__check_sensitivity(sensitivity):
                column['sensitivity'] = sensitivity

            if lower_bound is None: lower_bound = float('-inf')
            if upper_bound is None: upper_bound = float('inf')
            if self.__check_bounds(lower_bound, upper_bound):
                column['lower_bound'] = lower_bound
                column['upper_bound'] = upper_bound

            if round is not None:
                if not isinstance(round, int) or round < 0:
                    raise TypeError("round must be positive integer")
                else:
                    column['round'] = round

        if category is 'boolean':
            if self.__check_labels(self.df, column_name, label1, label2):
                column['label1'] = label1
                column['label2'] = label2

        self.__columns[str(column_name)] = column

    def execute(self, mode='normal'):

        laplace = LaplaceTruncated()
        binary = Binary()

        for column_name, details in self.__columns.items():

            if details['category'] is 'numeric':

                self.df[column_name] = pd.to_numeric(self.df[column_name], errors='coerce')

                laplace.set_epsilon_delta(epsilon=details['epsilon'], delta=details['delta'])
                laplace.set_sensitivity(details['sensitivity'])
                laplace.set_bounds(lower=details['lower_bound'], upper=details['upper_bound'])

                if 'round' in details:
                    round_randomise = lambda cell: round(laplace.randomise(cell), details['round'])

                    if mode is 'normal':
                        self.df[column_name] = self.df[column_name].apply(round_randomise)
                    elif mode is 'heavy':
                        self.df[column_name] = self.df[column_name].swifter.apply(round_randomise)

                else:
                    if mode is 'normal':
                        self.df[column_name] = self.df[column_name].apply(laplace.randomise)
                    elif mode is 'heavy':
                        self.df[column_name] = self.df[column_name].swifter.apply(laplace.randomise)

            elif details['category'] is 'boolean':

                self.df[column_name] = self.df[column_name].astype(str)

                binary.set_epsilon_delta(epsilon=details['epsilon'], delta=details['delta'])
                binary.set_labels(value0=details['label1'], value1=details['label2'])

                if mode is 'normal':
                    self.df[column_name] = self.df[column_name].apply(binary.randomise)
                elif mode is 'heavy':
                    self.df[column_name] = self.df[column_name].swifter.apply(binary.randomise)