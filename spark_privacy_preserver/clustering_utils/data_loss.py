class Dataloss:

    @staticmethod
    def complete_data_loss(df,factor = 20):
        clusters = df.groupby('cluster_number')
        data_loss = clusters.apply(lambda cluster: cluster_data_loss(cluster)).sum(axis=1).sum()
        total_loss = factor*len(df)*len(NUM_COL) + 10*len(df)*len(CAT_COL)
        return data_loss/total_loss


def cluster_data_loss(cluster,factor=20):
    num_cols = cluster[NUM_COL]
    cat_cols = cluster[CAT_COL]
    num_loss = abs((num_cols.max()-num_cols.min())/NUM_COL_RANGE*factor).sum()
    cat_loss = cat_col_data_loss(cat_cols.loc[cat_cols.index[0]])*len(cluster)
    return cat_loss+num_loss

def cat_col_data_loss(column):
    temp = (column.str.split(',').str.len())/CAT_COL_RANGE*10
    return temp