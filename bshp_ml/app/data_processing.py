
import logging
import asyncio
import zipfile
import os
import pandas as pd
import json
from datetime import datetime, timezone
from pydantic import TypeAdapter, ValidationError
# from pydantic_core import InitErrorDetails
# from typing import List
# from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
# import asyncio
# import uuid 

from tasks import task_manager
from settings import TEMP_FOLDER, USE_DETAILED_LOG
from api_types import DataRow
from db import db_processor

logging.getLogger("vbm_data_processing_logger").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class DataLoader:

    def __init__(self):
        pass

    async def upload_data_from_file(self, task):
        if USE_DETAILED_LOG:
            logger.info("saving data to temp zip file")

        await task_manager.update_task(task.task_id, status="UNZIPPING _DATA", progress=10)        
        folder = os.path.join(TEMP_FOLDER, task.task_id)
        os.makedirs(folder)
        if USE_DETAILED_LOG:
            logger.info("reading  data from zip file, unzipping")
        await self.get_data_from_zipfile(task.file_path, folder)

        zip_filename = os.path.basename(task.file_path)
        zip_filename_without_ext = os.path.splitext(zip_filename)[0]
        data_file_path = os.path.join(folder, f"{zip_filename_without_ext}.json")

        with open(data_file_path, 'r', encoding='utf-8-sig') as fp:
            json_data = json.load(fp)
        if USE_DETAILED_LOG:
            logger.info("validatind uploaded data")
        await task_manager.update_task(task.task_id, status="VALIDATING _DATA", progress=20)   

        data = []
        for row in json_data:
            data_row = DataRow.model_validate(row).model_dump() 
            data.append(data_row)     

        pd_data = pd.DataFrame(data) 
        pd_data['base_name'] = task.base_name
        pd_data['uploading_date'] = datetime.now(tz=timezone.utc)

        data = pd_data.to_dict(orient='records')
        if USE_DETAILED_LOG:
            logger.info("writing data to db")
        await task_manager.update_task(task.task_id, status="WRITING _TO_DB", progress=60)   

        if task.replace:
            await db_processor.delete_many('raw_data')

        await db_processor.insert_many('raw_data', data)

        await task_manager.cleanup_task_files(task.task_id)        

        return data

    async def get_data_from_zipfile(self, zip_file_path, folder):

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(folder)

    async def delete_data(self, db_filter=None):

        await db_processor.delete_many('raw_data',  db_filter=db_filter)

    async def get_data_count(self,  accounting_db='', db_filter=None):

        result = await db_processor.get_count('raw_data',  db_filter=db_filter)
        return result
    
    def get_none_data_row(self, parameters):
        row = {}
        for col in parameters['x_columns'] + parameters['y_columns']:
            if col in parameters['float_columns']:
                row[col] = 0
            elif col in parameters['str_columns']:
                row[col] = 'None'
            elif col in parameters['bool_columns']:
                row[col] = False                
            else:
                row[col] = None

        return pd.DataFrame([row])



class Reader:
    
    async def read(self, data_filter):
        if USE_DETAILED_LOG:
            logger.info("Start reading data")
        data = await db_processor.find("raw_data", convert_dates_in_db_filter(data_filter))
        pd_data = pd.DataFrame(data) 
        if USE_DETAILED_LOG:
            logger.info("Reading data. Done")        

        return pd_data


class Checker(BaseEstimator, TransformerMixin):

    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict

    def fit(self, X, y=None):
        return self
    
    def transform(self, X: pd.DataFrame):
        if USE_DETAILED_LOG:
            logger.info("Start checking data")
        if X.empty:
            raise ValueError('Fitting dataset is empty. Load more data or change filter.')
        if USE_DETAILED_LOG:
            logger.info("Checking data. Done")        
        return X


class DataEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict
        self.x_columns = self.parameters['x_columns']
        self.y_columns = self.parameters['y_columns']
        self.additional_columns = self.parameters['additional_columns']
        self.columns_to_encode = self.parameters['columns_to_encode']

        self.encode_dict = {}
        self.decode_dict = {}

    def fit(self, X, y=None):        
        if USE_DETAILED_LOG:
            logger.info("Start encoding data")
            logger.info("Form encode dict")
        
        for col in self.columns_to_encode:
            uniq = list(X[col].unique())

            uniq = [el for el in uniq if el]
            
            enc_dict = dict(zip(uniq, range(len(uniq))))
            self.encode_dict[col] = enc_dict

        return self
    
    def transform(self, X):
        if USE_DETAILED_LOG:
            logger.info("Encoding")
        for col in self.columns_to_encode:
            X[col] = X[col].apply(lambda x: self._get_encoded_field(x, col))

        X = X[self.additional_columns + self.x_columns + self.y_columns]
        if USE_DETAILED_LOG:
            logger.info("Encoding data. Done")  
        return X

    def inverse_transform(self, X):
        if USE_DETAILED_LOG:        
            logger.info("Start decoding data")        
        self.decode_dict = {}
        for col in self.encode_dict:
            d = {v: k for k, v in self.encode_dict[col].items()}
            self.decode_dict[col] = d

        for col in self.columns_to_encode:
            X[col] = X[col].apply(lambda x: self._get_decoded_field(x, col))
        if USE_DETAILED_LOG:
            logger.info("Decoding data. Done")  
        return X

    def _get_decoded_field(self, value, field):
        if value == -1:
            return ''
        else:
            return self.decode_dict[field][value]
    
    def _get_encoded_field(self, value, field):
        if not value:
            return -1
        else:
            return self.encode_dict[field][value]


class FeatureAdder(BaseEstimator, TransformerMixin):

    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict
        
        self.x_columns = self.parameters['x_columns']
        self.y_columns = self.parameters['y_columns']
        self.additional_columns = self.parameters['additional_columns']
        self.str_columns = self.parameters['str_columns']
        self.float_columns = self.parameters['float_columns']


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if USE_DETAILED_LOG:
            logger.info("Feature adding")

        X['document_year'] = X['date'].apply(self._get_year)
        X['document_month'] = X['date'].apply(self._get_month)

        X = X[self.additional_columns + self.x_columns + self.y_columns].copy()
        if USE_DETAILED_LOG:        
            logger.info("Adding features. Done")  
        return X
    
    def _get_month(self, date_value: datetime):
        return date_value.month

    def _get_year(self, date_value: datetime):
        return date_value.year


class NanProcessor(BaseEstimator, TransformerMixin):
    """ Transformer for working with nan values (deletes nan rows, columns, fills 0 to na values) """

    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict

        self.str_columns = self.parameters['str_columns']
        self.float_columns = self.parameters['float_columns']


    def fit(self, X, y=None):
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """
        Process nan values: removes all nan rows and columns, fills 0 instead single nan values
        :param x: data before nan processing
        :return: data after na  processing
        """
        if USE_DETAILED_LOG:        
            logger.info("Start processing Nan values") 
        x[self.str_columns] = x[self.str_columns].fillna('')
        x[self.float_columns] = x[self.float_columns].fillna(0)
        x.loc[x['year'] == '', 'year'] = '0'
        if USE_DETAILED_LOG:        
            logger.info("Processing Nan values. Done") 
        return x


class Shuffler(BaseEstimator, TransformerMixin):
    """
    Transformer class to shuffle data rows
    """
    def __init__(self, parameters, for_predict=False):
        self.parameters = parameters
        self.for_predict = for_predict        

    def fit(self, X, y=None):
        return self
    
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if USE_DETAILED_LOG:
            logger.info("Start data shuffling") 
        result = x.sample(frac=1).reset_index(drop=True).copy()
        if USE_DETAILED_LOG:        
            logger.info("Data shuffling. Done") 
        return result


def convert_dates_in_db_filter(db_filter, is_period=False):
    if isinstance(db_filter, list):
        result = []
        for el in db_filter:
            result.append(convert_dates_in_db_filter(el, is_period))
    elif isinstance(db_filter, dict):
        result = {}
        for k, v in db_filter.items():
            if k=='period':
                result[k] = convert_dates_in_db_filter(v, True)        
            else:
                result[k] = convert_dates_in_db_filter(v, is_period)
    elif isinstance(db_filter, str) and is_period:
        result = datetime.strptime(db_filter, '%d.%m.%Y')
    else:
        result = db_filter

    return result


data_loader = DataLoader()

