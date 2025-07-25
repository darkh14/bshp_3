import pandas as pd
import re
from _to_delete_db_connectors.connector import BaseConnector
import random
import logging
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def prepare_to_fit(df: pd.DataFrame, target_name: str):
    if target_name == 'group':
        x = df.drop(columns=[target_name, 'article_cash_flow', 'details_cash_flow', 'year'], axis=1)
        y = df[target_name]
    elif target_name == 'article_cash_flow':
        x = df.drop(columns=[target_name, 'details_cash_flow', 'year'], axis=1)
        y = df[target_name]
    elif target_name == 'details_cash_flow':
        x = df.drop(columns=[target_name, 'year'], axis=1)
        y = df[target_name]
    elif target_name == 'year':
        x = df.drop(columns=[target_name], axis=1)
        y = df[target_name]
    return x, y

    
def encode_objects_fit(df: pd.DataFrame):
    target_dict = {}
    df.reverse = df.reverse.astype('str')
    list_cols = ['article_cash_flow', 'details_cash_flow', 'year', 'moving_type', 'base_article','operation_type',  
                 'payment', 'reverse', 'type_of_customer', 'type_of_contract', 'account', 'sub_account', 'calculation_account', 'calculation_account_turnover', 
                 'calculation_account_total', 'account_kredit', 'account_debet', 'account_debet_turnover', 'account_debet_total', 'name_noom', 'type_noom', 
                 'unit_noom','view_noom', 'group_noom', 'group']
    for col in list_cols:
        details = df[col].unique()
        numbers = [i for i in range(len(details))]
        details_dict = dict(zip(details, numbers))
        target_dict[col] = details_dict
        df[col] = df[col].apply(lambda x: details_dict[x])
        df[col] = df[col].astype('int')
    logger.info("ENCODE DATA---------DONE")
    return df, target_dict
    
def principal_period(s: str):
    i = (s + s).find(s, 1, -1)
    return s if i == -1 else s[:i]
    
    
def change_name(s: str):
    if s !='':
        pattern = re.compile(r'\w+')
        result = pattern.findall(s)[0]
        s = ' '.join(result) 
    return s
    
    
def change_payment(s: str):
    if s !='':
        pattern = re.compile(r'\w+')
        result = pattern.findall(s)[0:4]
        s = ' '.join(result) 
    return s
    
    
def tramsform_data(df: pd.DataFrame):
    start_time = time.time()
    df['date'] = df['date'].apply(lambda x: int(x[3:5]))
    df.drop_duplicates(inplace=True, ignore_index=True)
    df.drop(['document', 'base_name', 'unit_of_count', 'is_service'], axis=1, inplace=True)
    if 'name_of_main_asset' in df.columns:
        df.drop(['name_of_main_asset', 'type_of_main_asset', 'group_of_main_asset'], axis=1, inplace=True)
    df.reverse = df.reverse.astype('str')
    df = df.fillna(0)
    df['name_noom'] = df['name_of_noomenclature'].map(str) + df['name_of_noomenclature_sub'].map(str) 
    df.drop(['name_of_noomenclature', 'name_of_noomenclature_sub'], axis=1, inplace=True)
    df['name_noom'] = df['name_noom'].apply(principal_period)

    df['type_noom'] = df['type_of_noomenclature'].map(str) + df['type_of_noomenclature_sub'].map(str)
    df.drop(['type_of_noomenclature', 'type_of_noomenclature_sub'], axis=1, inplace=True)
    df['type_noom'] = df['type_noom'].apply(principal_period)

    df['unit_noom'] = df['noomenclature_unit'].map(str) + df['noomenclature_unit_sub'].map(str)
    df.drop(['noomenclature_unit', 'noomenclature_unit_sub'], axis=1, inplace=True)
    df['unit_noom'] = df['unit_noom'].apply(principal_period)

    df['view_noom'] = df['view_of_noomenclature'].map(str) + df['view_of_noomenclature_sub'].map(str)
    df.drop(['view_of_noomenclature', 'view_of_noomenclature_sub'], axis=1, inplace=True)
    df['view_noom'] = df['view_noom'].apply(principal_period)

    df['group_noom'] = df['group_of_noomenclature'].map(str) + df['group_of_noomenclature_sub'].map(str)
    df.drop(['group_of_noomenclature', 'group_of_noomenclature_sub'], axis=1, inplace=True)
    df['group_noom'] = df['group_noom'].apply(principal_period)
  
    logger.info("MERGE COLUMNS--------DONE")
    end_time = time.time()
    print('MERGE COLUMNS time: ', end_time - start_time)
        
    df['name_noom'] = df['name_noom'].apply(change_name)
    df.loc[df['payment'] == 0, 'payment'] = ''
    df['payment'] = df['payment'].apply(change_payment)
    end_time = time.time()
    logger.info("TRANSFORM DATA--------DONE")
    print('TRANSFORM DATA time: ', end_time - start_time)
        
    return df
    
def transform_to_predict(db_connector: BaseConnector, df: pd.DataFrame):
    start_time = time.time()
    df_copy = df.copy()
    df_transform = tramsform_data(df)
    df_transform['document'] = df_copy['document']
    list_cols = ['moving_type', 'base_article','operation_type',  
                 'payment', 'reverse', 'type_of_customer', 'type_of_contract', 'account', 'sub_account', 'calculation_account', 'calculation_account_turnover', 
                 'calculation_account_total', 'account_kredit', 'account_debet', 'account_debet_turnover', 'account_debet_total', 'name_noom', 'type_noom', 
                 'unit_noom','view_noom', 'group_noom']
    for col in list_cols:
        result = db_connector.get_lines(col)[0]
        for row in range(len(df_transform)):
            if df_transform[col][row] in result.keys():
                df_transform.loc[row, col] = result[df_transform[col][row]]
            else:
                new_key = random.choice(list(result.keys()))
                df_transform.loc[row, col] = result[new_key]
    logger.info("TRANSFORM TO PREDICT--------DONE")
    end_time = time.time()
    print('TRANSFORM TO PREDICT time: ', end_time - start_time)
    return df_transform


def decode_objects(db_connector: BaseConnector, target: str, predicts: list):
    start_time = time.time()
    names_dict = db_connector.get_lines(target)[0]
    result_list = []
    for i in range(len(predicts)):
        for key in names_dict.keys():
            if predicts[i] == names_dict[key]:
                result_list.append(key)
    logger.info("DECODE--------DONE")
    end_time = time.time()
    print('DECODE time: ', end_time - start_time)
    return result_list