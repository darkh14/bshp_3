from abc import ABC, abstractmethod
import logging
from typing import Optional
from enum import Enum
import pickle
from datetime import datetime, UTC
import zlib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from data_processing import Reader, Checker, DataEncoder, NanProcessor, Shuffler
from db import db_processor
from api_types import ModelStatuses

logging.getLogger("bshp_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class Model(ABC):
    model_type = ''

    def __init__(self, base_name):
        self.base_name = base_name
        self.x_columns = ['is_reverse',
                    'document_month',
                    'document_year',
                    'moving_type',
                    'company_inn',
                    'company_kpp',
                    'base_document_kind',
                    'base_document_operation_type',
                    'contractor_name',
                    'contractor_inn',
                    'contractor_kpp',
                    'contractor_kind',
                    'article_name',
                    'is_main_asset',
                    'analytic',
                    'analytic2',
                    'analytic3',
                    'article_parent',
                    'article_group',
                    'article_kind',
                    'store',
                    'department',
                    'company_account_number',
                    'contractor_account_number',
                    'qty',
                    'price',
                    'sum']
        self.y_columns = ['cash_flow_item_code',
                          'cash_flow_details_code',
                          'year']  
        
        self.str_columns = ['moving_type',
                          'company_inn',
                          'company_kpp',
                          'base_document_kind',
                          'base_document_operation_type',
                          'contractor_name',
                          'contractor_inn',
                          'contractor_kpp',
                          'contractor_kind',
                          'article_name',
                          'analytic',
                          'analytic2',
                          'analytic3',
                          'article_parent',
                          'article_group',
                          'article_kind',
                          'store',
                          'department',
                          'company_account_number',
                          'contractor_account_number',
                          'cash_flow_item_code',
                          'year', 
                          'cash_flow_details_code']
        self.float_columns = ['qty',
                              'price',
                              'sum',]        
        self.parameters = {'x_columns': self.x_columns, 
                           'y_columns': self.y_columns, 
                           'str_columns': self.str_columns,
                           'float_columns': self.float_columns}
        
        self.status = ModelStatuses.CREATED
        self.error_text = ''

        self.fitting_start_date: Optional[datetime] = None
        self.fitting_end_date: Optional[datetime] = None

        self.metrics = {}

    @abstractmethod
    async def fit(self, parameters):
        ...
    
    @abstractmethod
    async def predict(self, x):
        ...    


class RfModel(Model):
    model_type = 'rf'
    def __init__(self, base_name):
        super().__init__(base_name)

        self.columns_to_encode = ['is_reverse',
                          'moving_type',
                          'company_inn',
                          'company_kpp',
                          'base_document_kind',
                          'base_document_operation_type',
                          'contractor_name',
                          'contractor_inn',
                          'contractor_kpp',
                          'contractor_kind',
                          'article_name',
                          'is_main_asset',
                          'analytic',
                          'analytic2',
                          'analytic3',
                          'article_parent',
                          'article_group',
                          'article_kind',
                          'store',
                          'department',
                          'company_account_number',
                          'contractor_account_number',
                          'cash_flow_item_code',
                          'year', 
                          'cash_flow_details_code']
        self.parameters['columns_to_encode'] = self.columns_to_encode                
        self.encode_dict = {}
        self.decode_dict = {}
        self.parameters['encode_dict'] = self.encode_dict
        self.parameters['decode_dict'] = self.decode_dict

        self.field_models = {} 
        self.data_encoder = None       

    async def fit(self, parameters):
        self.status = ModelStatuses.FITTING
        self.fitting_start_date = datetime.now(UTC)
        try:
            data_filter = parameters['data_filter']
            logger.info("Reading data from db")
            X_y = await Reader().read(data_filter)
            logger.info("Transforming and checking data")
            pipeline = Pipeline([
                                ('checker', Checker(self.parameters)),
                                ('nan_processor', NanProcessor(self.parameters)),                            
                                ('data_encoder', DataEncoder(self.parameters)),
                                ('shuffler', Shuffler(self.parameters)),
                                ])

            X_y = pipeline.fit_transform(X_y, [])
            c_x_columns= self.x_columns
            
            for ind, y_col in enumerate(self.y_columns):
                c_y_columns = [y_col]

                X, y = X_y[c_x_columns].to_numpy(), X_y[c_y_columns].to_numpy().ravel()
                logger.info('Start Fitting model. Field = "{}"'.format(y_col))

                model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=1, min_samples_split=5)
                model.fit(X, y)

                self.field_models[y_col] = model

                logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

                c_x_columns = c_x_columns + c_y_columns

            self.data_encoder = pipeline.named_steps['data_encoder']

            self.status = ModelStatuses.READY
            self.fitting_end_date = datetime.now(UTC)            
        except Exception as e:
            self.status = ModelStatuses.ERROR
            self.error_text = str(e)
            raise e

    async def predict(self, X, need_to_encode=True):
        
        logger.info("Transforming and checking data")
        X = pd.DataFrame(X)
        X_result = X.copy()

        pipeline_list = [
                        ('checker', Checker(self.parameters, for_predict=True)),
                        ('nan_processor', NanProcessor(self.parameters, for_predict=True)),                            
                        ]
        
        if need_to_encode:
            pipeline_list.append(('data_encoder', self.data_encoder))

        pipeline = Pipeline(pipeline_list)

        X_y = pipeline.transform(X).copy()
        c_x_columns= self.x_columns
        
        for ind, y_col in enumerate(self.y_columns):
            c_y_columns = [y_col]

            X = X_y[c_x_columns].to_numpy()
            logger.info('Start predicting. Field = "{}"'.format(y_col))

            model = self.field_models[y_col]
            y = model.predict(X)

            X_y[y_col] = y

            logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

            c_x_columns = c_x_columns + c_y_columns  

        if need_to_encode:
            X_y = pipeline.named_steps['data_encoder'].inverse_transform(X_y)

        for col in self.y_columns:
            X_result[col] = X_y[col]

        return X_result.to_dict(orient='records')       
    


class ModelManager:

    def __init__(self):
        self.models = []

    async def read_models(self):
        models = await db_processor.find('models')
        for model_row in models:
            compressed_bin = model_row['binary']
            model_bin = zlib.decompress(compressed_bin)
            model_row['model'] = pickle.loads(model_bin)
            del model_row['binary']

        self.models = models

    async def write_model(self, model: Model):
        if model.status != ModelStatuses.READY:
            raise ValueError('Model is not ready to be written to db')

        model_bin = pickle.dumps(model)
        compressed_model_bin = zlib.compress(model_bin, level=9)

        
        await db_processor.insert_one('models', 
                                      {'model_type': model.model_type, 'base_name': model.base_name, 'binary': compressed_model_bin}, 
                                      {'model_type': model.model_type, 'base_name': model.base_name})

    def get_model(self, model_type='rf', base_name=''):

        model_list = [el for el in self.models if el['model_type'] == model_type and el['base_name'] == base_name]
        if model_list:
            model = model_list[0]['model']
        else:
            model = RfModel(base_name=base_name)

        return model
    
    async def delete_model(self, model_type='rf', base_name=''):

        self.models = [el for el in self.models if el['model_type'] != model_type and el['base_name'] != base_name]
        await db_processor.delete_many('models', db_filter={'model_type': model_type, 'base_name': base_name})

    async def get_info(self, model_type='rf', base_name=''):

        model = self.get_model(model_type=model_type, base_name=base_name)

        return {'status': model.status, 
                'error_text': model.error_text, 
                'fitting_start_date': model.fitting_start_date, 
                'fitting_end_date': model.fitting_end_date,
                'metrics': model.metrics}        
    

model_manager = ModelManager()