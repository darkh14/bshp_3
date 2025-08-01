from abc import ABC, abstractmethod
import logging
from typing import Optional
from enum import Enum
import pickle
from datetime import datetime, UTC
import zlib
import os
import uuid
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from data_processing import Reader, Checker, DataEncoder, NanProcessor, Shuffler
from db import db_processor
from api_types import ModelStatuses, ModelTypes
from settings import MODEL_FOLDER

logging.getLogger("bshp_data_processing_logger").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class Model(ABC):
    model_type = None

    def __init__(self, base_name):
        self.base_name = base_name

        self.uid = str(uuid.uuid5(uuid.NAMESPACE_DNS, '{}_{}'.format(self.model_type.value if self.model_type else '', self.base_name)))
        self.x_columns = [
                    'is_reverse',
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
        self.y_columns = [
                          'cash_flow_item_code',
                          'cash_flow_details_code',
                          'year'
                          ]  
        
        self.additional_columns = ['number', 'date',]

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
                           'float_columns': self.float_columns,
                           'additional_columns': self.additional_columns}
        
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
    model_type = ModelTypes.rf
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

        self.__init__(self.base_name)

        self.field_models = {}
        try:
            data_filter = parameters['data_filter']
            logger.info("Reading data from db")
            X_y = await Reader().read(data_filter)

            train_indexes, test_indexes = self._get_train_test_indexes(X_y)

            logger.info("Transforming and checking data")
            pipeline = Pipeline([
                                ('checker', Checker(self.parameters)),
                                ('nan_processor', NanProcessor(self.parameters)),                            
                                ('data_encoder', DataEncoder(self.parameters)),
                                ('shuffler', Shuffler(self.parameters)),
                                ])

            X_y = pipeline.fit_transform(X_y, [])

            X_y_train, X_y_test = self._get_train_test_datasets(X_y, train_indexes, test_indexes)

            c_x_columns= self.x_columns

            test_field_models = {}
            
            for y_col in self.y_columns:
                c_y_columns = [y_col]

                X, y = X_y[c_x_columns].to_numpy(), X_y[c_y_columns].to_numpy().ravel()
                X_train, y_train = X_y_train[c_x_columns].to_numpy(), X_y_train[c_y_columns].to_numpy().ravel()

                logger.info('Start Fitting model. Field = "{}"'.format(y_col))

                model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                model_test = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)

                model.fit(X, y)
                model_test.fit(X_train, y_train)

                self.field_models[y_col] = model
                test_field_models[y_col] = model_test                

                logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

                c_x_columns = c_x_columns + c_y_columns

            self.data_encoder = pipeline.named_steps['data_encoder']

            datasets = {'all': X_y, 'train': X_y_train, 'test': X_y_test}
            models = {'all': self.field_models, 'train': test_field_models, 'test': test_field_models}

            logger.info("Start calculating metrics")
            self.metrics = self._get_metrics(datasets, models) 
            logger.info("Calculating metrics. Done")                     

            self.status = ModelStatuses.READY
            self.fitting_end_date = datetime.now(UTC)            
        except Exception as e:
            self.status = ModelStatuses.ERROR
            self.error_text = str(e)
            raise e

    def _get_metrics(self, datasets, models):
        # predictions to calculate metrics
        logger.info('Get metrics')       
        metrics = {}
           
        for name, dataset in datasets.items():
            c_x_columns = self.x_columns 
            dataset['check'] = 0

            for y_col in self.y_columns:
                logger.info('Calculating metrics dataset "{}", field "{}"'.format(name, y_col)) 
                c_X = dataset[c_x_columns].to_numpy()

                field_models = models[name]
                c_y_pred = field_models[y_col].predict(c_X)

                c_y_column = '{}_pred'.format(y_col)
                check_column = '{}_check'.format(y_col)

                dataset[c_y_column] = c_y_pred
                dataset[check_column] = dataset[y_col] != dataset[c_y_column] 
                dataset[check_column] = dataset[check_column].astype(int)

                c_x_columns = c_x_columns + [c_y_column]

                dataset['check'] = dataset['check'] + dataset[check_column]
                logger.info('Done')

            dataset['check'] = dataset['check'] == 0
            dataset['check'] = dataset['check'].astype(int)

            dataset['row'] = 1

            dataset_grouped = dataset[['number', 'date', 'row', 'check']].groupby(by=['number', 'date']).sum()
            dataset_grouped['check_all'] = dataset_grouped['row'] == dataset_grouped['check']
            dataset_grouped['check_all'] = dataset_grouped['check_all'].astype(int)

            all = dataset_grouped['check_all'].shape[0]
            right = dataset_grouped[dataset_grouped['check_all']==1].shape[0]
            metrics['acuracy_{}'.format(name)] = right/all

    async def predict(self, X, need_to_encode=True):
        
        if self.status != ModelStatuses.READY:
            raise ValueError('Model is not ready. Fit it before.')

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
        
        for y_col in self.y_columns:
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
    
    def _get_train_test_indexes(self, X_y):
        indexes_len = X_y.shape[0]
        indexes = np.arange(indexes_len)

        test_size=0.2

        train_indexes = indexes[:int(indexes_len*(1-test_size))]
        test_indexes = indexes[int(indexes_len*(1-test_size)):]

        return train_indexes, test_indexes

    def _get_train_test_datasets(self, X_y, train_indexes, test_indexes):

        train_dataset = X_y.iloc[train_indexes].copy()
        test_dataset = X_y.iloc[test_indexes].copy()

        return train_dataset, test_dataset                


class ModelManager:

    def __init__(self):
        self.models = []

    async def read_models(self):

        models = []
        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)
        
        model_files = os.listdir(MODEL_FOLDER)

        for model_file in model_files:
            ext = os.path.splitext(model_file)[-1]

            if ext != '.mdl':
                continue
            model = None
            try:
                with open(os.path.join(MODEL_FOLDER, model_file), 'rb') as fp:
                    model_bin_compressed = fp.read()
                    model_bin = zlib.decompress(model_bin_compressed)
                    model = pickle.loads(model_bin)

            except Exception as e:
                pass

            if model:
                models.append({'model_type': model.model_type, 'base_name': model.base_name, 'model': model})

        self.models = models

    async def write_model(self, model):

        model_list = [el for el in self.models if el['model_type'] == model.model_type and el['base_name'] == model.base_name]

        if not model_list:
            self.models.append({'model_type': model.model_type, 'base_name': model.base_name, 'model': model})

        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)  

        model_file = os.path.join(MODEL_FOLDER, '{}.mdl'.format(model.uid)) 
        with open(model_file, 'wb') as fp:
            model_bin = pickle.dumps(model)
            model_bin_compressed = zlib.compress(model_bin, level=9)
            fp.write(model_bin_compressed) 

    def get_model(self, model_type=ModelTypes.rf, base_name=''):

        model_list = [el for el in self.models if el['model_type'] == model_type and el['base_name'] == base_name]
        if model_list:
            model = model_list[0]['model']
        else:
            model = RfModel(base_name=base_name)

        return model
    
    async def delete_model(self, model_type=ModelTypes.rf, base_name=''):
        
        c_models = [el for el in self.models if el['model_type'] != model_type and el['base_name'] == base_name]
        if c_models:
            model = c_models[0]
        elif model_type == ModelTypes.rf:
            model = RfModel(base_name=base_name)
        else: 
            raise ValueError('Model type "{}" is not supported')

        self.models = [el for el in self.models if el['model_type'] != model_type and el['base_name'] != base_name]

        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)

        model_file = os.path.join(MODEL_FOLDER, '{}.mdl'.format(model.uid))
        if os.path.exists(model_file):
            os.remove(model_file)

    async def get_info(self, model_type=ModelTypes.rf, base_name=''):

        model = self.get_model(model_type=model_type, base_name=base_name)

        return {'status': model.status, 
                'error_text': model.error_text, 
                'fitting_start_date': model.fitting_start_date, 
                'fitting_end_date': model.fitting_end_date,
                'metrics': model.metrics}        



model_manager = ModelManager()