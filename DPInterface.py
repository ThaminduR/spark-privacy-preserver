import pandas as pd  # type: ignore
import pyspark  # type: ignore

from diffprivlib.mechanisms import LaplaceTruncated  # type: ignore
from diffprivlib.mechanisms import Binary  # type: ignore

import swifter  # type: ignore

# Following imports are purely to provide type checking functionality to the library
from pandas import DataFrame as PandasDataFrame  # type: ignore
from pyspark.sql.dataframe import DataFrame as SparkDataFrame
from pyspark.rdd import RDD
from numpy import ndarray  # type: ignore
from numbers import Real  # type: ignore
from typing import List, Dict, Union, Optional  # type: ignore
from typing import TypedDict  # type: ignore


class Column(TypedDict, total=False):
    column_name: str
    category: str
    epsilon: Union[int, float]
    delta: Union[int, float]
    sensitivity: Union[int, float, None]
    lower_bound: Union[int, float]
    upper_bound: Union[int, float]
    round: int
    label1: Optional[str]
    label2: Optional[str]


class DPInterface:
    r""" Create differentially private Pandas dataset

    The class makes use of the 'diffprivlib' library by IBM to create differentially private Pandas dataset.
    It can handle columns with two major data types: 'numeric' and 'boolean'
        'numeric': column should have only numbers or `NaN' value
                    utilizes LaplaceTruncated mechanism from diffprivlib.mechanisms

        'boolean': each row of column should have one of two boolean values set by user
                    utilizes Binary mechanism from diffprivlib.mechanisms

    Attributes:
        df: A Pandas DataFrame object. Methods such as `__check_labels`, `add_column` and `execute`
            will raise ValueError if df is None.

    Methods:
        set_global_epsilon_delta:
            Assigns common epsilon and delta to be used by all columns if they lack
            column specific values for epsilon and delta

        set_global_sensitivity:
            Assigns common sensitivity to be used by all columns if they lack
            column specific value for sensitivity

        set_df: Assigns DataFrame object to the class

        add_column: Adds column dictionary to `__columns`

        execute: changes df to be differentially private. This change is not reversible.

    Examples:
        foo = DPInterface()
            You may have to manually set epsilon for each column. Otherwise raises ValueError

        foo = DPInterface(global_epsilon=0.00001)

        foo = DPInterface(global_epsilon=0.00001, global_delta=0.5)
            global_delta is an optional parameter

    """

    def __init__(self,
                 global_epsilon: Union[int, float, None] = None,
                 global_delta: Union[int, float] = 0.0,
                 df: Optional[PandasDataFrame] = None) -> None:
        r""" Inits DPInterface with either epsilon, delta and df or None

        Args:
            global_epsilon: Common epsilon value to be used by all columns as a fail-safe
            global_delta: Common delta value to be used by all columns as a fail-safe. Defaults to 0.0
            df: DataFrame to be converted. Can change to a different dataframe with set_df()
        """

        self.df: Optional[PandasDataFrame] = df
        self.__columns: Dict[str, Column] = {}

        self.__epsilon: Optional[float] = None
        self.__delta: Optional[float] = None

        self.__sensitivity: Union[int, float, None] = None

        if global_epsilon is not None and self.__check_epsilon_delta(global_epsilon, global_delta):
            self.__epsilon = float(global_epsilon)
            self.__delta = float(global_delta)

    @staticmethod
    def __check_epsilon_delta(epsilon: Union[int, float], delta: Union[int, float]) -> bool:
        r""" checks whether epsilon and delta meet required conditions for method `execute`

        Always called disregarding category. Both 'LaplaceTruncated' and 'Binary' mechanism
        require epsilon and delta.

        Args:
            epsilon: epsilon value to be used by method `execute`
            delta: delta value to be used by method `execute`

        Returns: True if parameters satisfy the conditions

        Raises: TypeError, ValueError if parameters have not been set correctly

        """
        if not isinstance(epsilon, Real) or not isinstance(delta, Real):
            raise TypeError("Epsilon and delta must be numeric")

        if epsilon < 0:
            raise ValueError("Epsilon must be non-negative")

        if not 0 <= float(delta) <= 1:
            raise ValueError("Delta must be in range [0, 1]")

        if epsilon == 0 and delta == 0:
            raise ValueError("Epsilon and Delta cannot both be zero")

        return True

    @staticmethod
    def __check_sensitivity(sensitivity: Union[int, float, None]) -> bool:
        r""" checks whether sensitivity meets required conditions for method `execute`

        Called only when category = 'numeric'

        Args:
            sensitivity: sensitivity value to be used by method `execute`. Only apply for category 'numeric'

        Returns: True if parameter satisfies the conditions

        Raises: TypeError, ValueError if parameter has not been set correctly

        """
        if not isinstance(sensitivity, Real):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        return True

    @staticmethod
    def __check_labels(df: PandasDataFrame, column_name: str, label1: Optional[str], label2: Optional[str]) -> bool:
        r""" checks whether labels meet required conditions for method `execute`.

        Called only when category = 'boolean'

        Args:
            df: DataFrame object
            column_name: specific column which is to be executed using 'Binary' mechanism
            label1: label to be used by 'Binary' mechanism
            label2: label to be used by 'Binary' mechanism

        Returns: True if parameters satisfy the conditions

        Raises: TypeError, ValueError if parameters have not been set correctly

        """
        if not isinstance(label1, str) or not isinstance(label2, str):
            raise TypeError("Labels must be strings.")

        if len(label1) == 0 or len(label2) == 0:
            raise ValueError("Labels must be non-empty strings")

        if label1 == label2:
            raise ValueError("Labels must not match")

        labels: ndarray = df[column_name].unique()
        if len(labels) is not 2 or label1 not in labels or label2 not in labels:
            # checks whether all the rows of column have either label1 or label2
            raise ValueError("Column has multiple unique labels")

        return True

    @staticmethod
    def __check_bounds(lower: Union[int, float], upper: Union[int, float]) -> bool:
        r""" checks whether lower and upper bounds for a specific column bound to their conditions

        Called only when category = 'numeric'

        Args:
            lower: lower bound of a column
            upper: upper bound of a column

        Returns: True if parameters satisfy the conditions

        Raises: TypeError, ValueError if parameters have not been set correctly

        """

        if not isinstance(lower, Real) or not isinstance(upper, Real):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")

        return True

    def set_global_epsilon_delta(self, epsilon: Union[int, float], delta: Union[int, float] = 0.0) -> None:
        r"""  set global epsilon and delta if they satisfy method `__check_epsilon_delta`

        Args:
            epsilon: epsilon value to be used by method `execute`
            delta: delta value to be used by method `execute`

        Raises:
            The method itself will not raise any exceptions.
            However inner method ``__check_epsilon_delta` may raise Exception,
            only if parameters have not been set correctly
        """

        if self.__check_epsilon_delta(epsilon, delta):
            self.__epsilon = float(epsilon)
            self.__delta = float(delta)

    def set_global_sensitivity(self, sensitivity: Union[int, float]) -> None:
        r"""  set global sensitivity if it satisfies method `__check_sensitivity`

        Args:
            sensitivity: sensitivity value to be used by method `execute`.

        Raises:
            The method itself will not raise any exceptions.
            However inner method `__check_sensitivity` may raise Exception,
            only if parameters have not been set correctly
        """

        if self.__check_sensitivity(sensitivity):
            self.__sensitivity = float(sensitivity)

    def set_df(self, data: Union[PandasDataFrame, SparkDataFrame, str],
               engine: str = 'python',
               encoding: Optional[str] = None,
               header: Union[int, List[int], None] = None,
               names: List[str] = None) -> None:
        r""" sets DataFrame object to class.

        Multiple DataFrames can be set, but not at same time.

        Args:
            header: specific for csv file. performs same as header attribute in pandas.read_csv() method
            names: specific for csv file. sets column names for pandas columns.
            encoding: specific for csv file. sets encoding of csv file.
            engine: specific for csv file. can use different engines to import csv files. default: 'python'
            data: can be a Pandas DataFrame object, Spark DataFrame object, Spark RDD or CSV file

        """

        if isinstance(data, PandasDataFrame):
            data = data
        elif isinstance(data, SparkDataFrame):
            data = data.toPandas()

        # TODO: working on RDD
        elif isinstance(data, RDD):
            pass
        elif isinstance(data, str) and data[-4:] == '.csv':
            data = pd.read_csv(filepath_or_buffer=data, header=header, engine=engine, encoding=encoding, names=names)

        self.df = data

    def add_column(self, column_name: str, category: str,
                   epsilon: Union[int, float, None] = None, delta: Union[int, float, None] = None,
                   sensitivity: Union[int, float, None] = None,
                   lower_bound: Union[int, float, None] = None, upper_bound: Union[int, float, None] = None,
                   round: Optional[int] = None,
                   label1: Optional[str] = None, label2: Optional[str] = None) -> None:

        r""" adds a column with required parameters to the __columns dictionary.

        A column may have specific details. Hence this method allows user to set them individually for a column.
        However in case any one parameter is missing, appropriate value will be copied from available global
        parameters and passed instead.

        Args:
            -------------------- common arguments --------------------------
            column_name: Name of column
            category: Category the column belongs to: ['numeric', 'boolean']
            epsilon: Epsilon value to be used by method `execute`
            delta: Delta value to be used by method `execute`

            ---------------------- arguments specific to category = 'numeric' ----------------------
            sensitivity: Sensitivity value to be used by method `execute`.
            lower_bound: Lower bound of a column.
            upper_bound: Upper bound of a column.
            round: Rounding factor. Values can be rounded off after applying a certain mechanism.

            ---------------------- arguments specific to category = 'boolean' ----------------------
            label1: label to be used by 'Binary' mechanism.
            label2: label to be used by 'Binary' mechanism.

        Raises: TypeError, ValueError if parameters have not been set correctly.
                Inner methods may raise Exception.

        """

        if self.df is None:
            raise ValueError("Add an eligible DataFrame before adding columns")

        if column_name not in list(self.df.columns):
            raise ValueError("Cannot find column in given DataFrame")

        if category not in ['numeric', 'boolean']:
            raise ValueError("Cannot find category in available list")

        column: Column = {'category': category}

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

            if sensitivity is None:
                sensitivity = self.__sensitivity
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

    def execute(self, mode: Optional[str] = None):
        r"""
        Args:
            mode: mode to which method `execute` works. 'heavy' mode applies Swifter.

        Raises:
            The method itself will not raise any exceptions.
            However inner methods may raise Exception, only if parameters have not been set correctly

        """

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

                    if mode is 'heavy':
                        self.df[column_name] = self.df[column_name].swifter.apply(round_randomise)
                    else:
                        self.df[column_name] = self.df[column_name].apply(round_randomise)

                else:
                    if mode is 'heavy':
                        self.df[column_name] = self.df[column_name].swifter.apply(laplace.randomise)
                    else:
                        self.df[column_name] = self.df[column_name].apply(laplace.randomise)

            elif details['category'] is 'boolean':

                self.df[column_name] = self.df[column_name].astype(str)

                binary.set_epsilon_delta(epsilon=details['epsilon'], delta=details['delta'])
                binary.set_labels(value0=details['label1'], value1=details['label2'])

                if mode is 'normal':
                    self.df[column_name] = self.df[column_name].apply(binary.randomise)
                elif mode is 'heavy':
                    self.df[column_name] = self.df[column_name].swifter.apply(binary.randomise)
