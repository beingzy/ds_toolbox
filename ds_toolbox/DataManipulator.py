""" suites of functions for data manipulation/transformation

    Author: Yi Zhang <beingzy@gmail.com>
    Date: 2016/05/19
"""
from datetime import datetime
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


def convert_questionaire_to_dataframe(df, group_vars, column_var, value_var):
    """ convert question-answer per record dataframe to
        dataframe organized per entity, with questions as columns
    
    Parameters:
    ==========
    df: {pandas.DataFrame}
    group_vars: {list} list of column names
    column_var: string
    value_var: string 
    """
    def create_dict(keys, values):
        record = {}
        for key, val in zip(keys, values):
            record[key] = val
        return record
    
    start_dt = datetime.now()   
     
    grouped = df.groupby(group_vars)
    records = []
    pbar = tqdm(total=len(grouped))
    
    for name, group in grouped:
        record = {}
        for key, val in zip(group_vars, name):
            record[key] = val
        dict_content = create_dict(group[column_var], group[value_var])
        record.update(dict_content)
        records.append(record)
        
        pbar.update()
    
    # transform list of dictionary to DataFrame
    new_df = DataFrame(records)
    # re-order columns 
    new_colnames = [i for i in new_df.columns.values if not i in group_vars]
    new_df = new_df[group_vars + new_colnames]
    
    dt_cost = (datetime.now() - start_dt).total_seconds()
    print("operation time cost: {:.2f} mins".format(dt_cost/60))
    
    return new_df