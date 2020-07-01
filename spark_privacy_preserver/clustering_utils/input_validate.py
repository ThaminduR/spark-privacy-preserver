from .. import gv as gvv

class InputValidator:

    @staticmethod
    def validate_input(df,QI_attr,Sensitive_attr,cat_indecies,verbose,max_iter,anonimize_ratio,max_cluster_distance,nan_replacement_int=0,nan_replacement_str=''):
        QI_LEN,QI,_df,SA,IS_CAT,QI_RANGE_VAL,QI_RANGE_VAL,CAT_UNIQUE,NUM_COL,CAT_COL,_DEBUG,RANGE_FIX,CAT_INDEXES,NUM_COL_RANGE,CAT_COL_RANGE = validator(df,QI_attr,Sensitive_attr,cat_indecies,nan_replacement_int,nan_replacement_str)

        gv = [QI, SA, IS_CAT, QI_LEN, QI_RANGE_VAL, _df,NUM_COL,NUM_COL_RANGE,CAT_COL,CAT_COL_RANGE,CAT_INDEXES,RANGE_FIX]
        gvname = ['QI', 'SA', 'IS_CAT', 'QI_LEN', 'QI_RANGE_VAL', '_df','NUM_COL','NUM_COL_RANGE','CAT_COL','CAT_COL_RANGE','CAT_INDEXES','RANGE_FIX']
        gv_dict = {}
        for i in zip(gvname,gv):
            gv_dict[i[0]] = i[1]

        gvv.init(gv_dict)

    @staticmethod
    def L_Diverse_Validate(dataframe , quasi_identifiers , sensitive_attributes):
        err_msg_1 = "quasi identifiers are not in columns of the dataframe"
        err_msg_2 = "Sensitive Attributes are not in columns of the dataframe"
        err_msg_3 = "Sensitive Attributes cannot be a subset of quasi identifiers"
        err_msg_7 = "Categorical index list cannot be more than quasi identifiers"
        err_msg_4 = "Invalid Catergorical index"
        err_msg_5 = "Duplicate Value"
        err_msg_6 = "Expect argument as a list"
        columns = dataframe.columns

        if(type(quasi_identifiers) != list):
            raise AnonymizeError(message = err_msg_6+" in "+ 'quasi identifiers')
        elif(type(sensitive_attributes) != list):
            raise AnonymizeError(message = err_msg_6+" in "+ 'sensitive attributes')
        elif(len(set(quasi_identifiers)) != len(quasi_identifiers)):
            raise AnonymizeError(message = err_msg_5+" in "+'quasi identifiers')
        elif(len(set(sensitive_attributes)) != len(sensitive_attributes)):
            raise AnonymizeError(message = err_msg_5+" in "+ 'sensitive attributes')      
        elif not(set(quasi_identifiers).issubset(set(columns))):
            raise AnonymizeError(message = err_msg_1)
        elif not(set(sensitive_attributes).issubset(set(columns))):
            raise AnonymizeError(message = err_msg_2)
        elif(len(set(sensitive_attributes).intersection(set(quasi_identifiers))) > 0):
            raise AnonymizeError(message = err_msg_3)
        elif(len(sensitive_attributes) == 0):
            raise AnonymizeError(message = "Sensitive Attributes cannot be empty")
        return True


def validator(dataframe,QI_,SA_,CAT_INDEXES_,nan_replacement_int,nan_replacement_str):
    global QI_LEN,QI,_df,SA,IS_CAT,QI_RANGE_VAL,QI_RANGE_VAL,CAT_UNIQUE,NUM_COL,CAT_COL,_DEBUG,RANGE_FIX,CAT_INDEXES,NUM_COL_RANGE,CAT_COL_RANGE
    err_msg_1 = "quasi identifiers are not in columns of the dataframe"
    err_msg_2 = "Sensitive Attributes are not in columns of the dataframe"
    err_msg_3 = "Sensitive Attributes cannot be a subset of quasi identifiers"
    err_msg_7 = "Categorical index list cannot be more than quasi identifiers"
    err_msg_4 = "Invalid Catergorical index"
    err_msg_5 = "Duplicate Value"
    err_msg_6 = "Expect argument as a list"
    columns = dataframe.columns
    if(type(QI_) != list):
        raise AnonymizeError(message = err_msg_6+" in "+ 'quasi identifiers')
    elif(type(SA_) != list):
        raise AnonymizeError(message = err_msg_6+" in "+ 'sensitive attributes')
    elif(type(CAT_INDEXES_) != list):
        raise AnonymizeError(message = err_msg_6+" in "+ 'catergorical index')
    elif(len(set(QI_)) != len(QI_)):
        raise AnonymizeError(message = err_msg_5+" in "+'quasi identifiers')
    elif(len(set(SA_)) != len(SA_)):
        raise AnonymizeError(message = err_msg_5+" in "+ 'sensitive attributes')
      
    elif not(set(QI_).issubset(set(columns))):
        raise AnonymizeError(message = err_msg_1)
    elif not(set(SA_).issubset(set(columns))):
        raise AnonymizeError(message = err_msg_2)
    elif(len(set(SA_).intersection(set(QI_))) > 0):
        raise AnonymizeError(message = err_msg_3)
    else:
        try:
            CAT_INDEXES_ = list(map(int,CAT_INDEXES_))
        except:
            raise AnonymizeError(message = err_msg_4+"\n Index should be a integer")
        CAT_INDEXES_sorted = sorted(CAT_INDEXES_)
        if(len(CAT_INDEXES_) != 0):
            if not(0 <= CAT_INDEXES_sorted[-1] < len(QI_)):
              raise AnonymizeError(message = err_msg_4+"\ncatergorical index should start with zero and in between 0 and number of quasi identifiers")
            if not(0 <= CAT_INDEXES_sorted[0] < len(QI_)):
              raise AnonymizeError(message = err_msg_4+"\ncatergorical index should start with zero and in between 0 and number of quasi identifiers")
        if(len(CAT_INDEXES_) > len(QI_)):
            raise AnonymizeError(message = err_msg_7)
        elif(len(set(CAT_INDEXES_)) != len(CAT_INDEXES_)):
            raise AnonymizeError(message = err_msg_5 +" in "+'Categorical Index')
        IS_CAT = [False]*len(QI_)
        try:
            if(len(CAT_INDEXES_) != 0):
                for index in CAT_INDEXES_:
                    IS_CAT[index] = True
                print("\n")
                print(CAT_INDEXES_,IS_CAT)
                print("\n")
        except:
            raise AnonymizeError(message = "Invalid index for categorical indexes")
        NUM_COL_RANGE=[]
        CAT_COL_RANGE=[]
        QI = QI_
        SA = SA_
        QI_LEN,QI,_df,SA,IS_CAT,QI_RANGE_VAL,QI_RANGE_VAL,CAT_UNIQUE,NUM_COL,CAT_COL,_DEBUG,RANGE_FIX,CAT_INDEXES = marking_globals(dataframe)
        _df = df_validator(_df,nan_replacement_int,nan_replacement_str)
        for i in range(QI_LEN):
            if(IS_CAT[i] is False):
                diff = _df[QI[i]].max() - _df[QI[i]].min()
                NUM_COL_RANGE.append(diff)
            else:
                CAT_COL_RANGE.append(len(_df[QI[i]].unique()))
        return QI_LEN,QI,_df,SA,IS_CAT,QI_RANGE_VAL,QI_RANGE_VAL,CAT_UNIQUE,NUM_COL,CAT_COL,_DEBUG,RANGE_FIX,CAT_INDEXES,NUM_COL_RANGE,CAT_COL_RANGE

def df_validator(df,nan_replacement_int=0,nan_replacement_str=''):
    df = df.dropna(how='all')
    df[CAT_COL] = df[CAT_COL].applymap(str).fillna(nan_replacement_str)
    df[CAT_COL] = df[CAT_COL].applymap(lambda x: x.replace(",","/"))
    df[NUM_COL] = df[NUM_COL].fillna(nan_replacement_int).applymap(lambda x: numerical_validator(x,nan_replacement_int))
    return df


def numerical_validator(value,nan_replace_int):
    try:
        if(type(value) == int or type(value) == float):
            return value
        elif('-' in str(value) or '- ' in str(value) or ' -' in str(value)):
            elements = value.split('-')
            return (int(elements[0].strip()) + int(elements[1].strip()))//2
        else:
            return int(value)
    except ValueError:
        return nan_replace_int
    except Exception as e:
        if(_DEBUG):
            print(e)                                                              ################################################
        return nan_replace_int

def marking_globals(df):
    QI_LEN = len(QI)
    CAT_INDEXES,CAT_COL,NUM_COL = [],[],[]
    drop_col = []
    CAT_UNIQUE = []
    _DEBUG = True
    QI_RANGE_VAL = []
    RANGE_FIX = 1
    if(_DEBUG):
      print("Starting initializing globals")
    for column in df.columns:
        if not((column in QI) | (column in SA)):
            drop_col.append(column)
    if(_DEBUG):
      print("After initializing drop column")
    df = df.drop(drop_col, axis=1)
    # df = df[QI+SA]
    _df = df.loc[:,QI+SA]
    sensitive_input = df.loc[:,SA]
    if(_DEBUG):
      print("Before marking QI_RANGE_VAL")
    try:
      for i in range(QI_LEN):
          if(IS_CAT[i] is False):
              diff = df[QI[i]].max() - df[QI[i]].min()
              QI_RANGE_VAL.append(diff)
          else:
            unique_count = len(df[QI[i]].unique())
            CAT_UNIQUE.append(unique_count)
            QI_RANGE_VAL.append(unique_count)
    except:
      raise(AnonymizeError(message = "Invalid categorical index. Check whether catergorical indexes are correct"))
    if(_DEBUG):
      print("Before marking NUM_COL & CAT_COL & CAT_INDEXES")
    for i,ele in enumerate(IS_CAT):
        if(ele):
          CAT_COL.append(QI[i])
          CAT_INDEXES.append(i)
        else:
          NUM_COL.append(QI[i])
    if(_DEBUG):
        print("Finished initializing Globals")
    return QI_LEN,QI,_df,SA,IS_CAT,QI_RANGE_VAL,QI_RANGE_VAL,CAT_UNIQUE,NUM_COL,CAT_COL,_DEBUG,RANGE_FIX,CAT_INDEXES


class AnonymizeError(Exception):
    def __init__(self, message="Invalid Input"):
        self.message = message
        super().__init__(self.message)

