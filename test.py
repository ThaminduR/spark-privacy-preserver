import pandas as pd
from DPLib import DPInterface

df = pd.read_csv("data_53000kb.csv",
                 engine='python',
                 header=None,
                 encoding="ISO-8859-1",
                 names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])
print(df.head(20))


test = DPInterface(global_epsilon=0.00001, global_delta=0.5)
test.set_global_sensitivity(10)
test.set_df(df)

test.add_column('I', 'numeric', lower_bound=0.35, upper_bound=0.8)
test.execute()


print(df.head(20))
