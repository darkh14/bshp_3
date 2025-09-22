from abc import ABC, abstractmethod
import logging
from typing import Optional
from enum import Enum
import pickle
from datetime import datetime, UTC
import os
import uuid
import pandas as pd
import numpy as np

import gc

import shutil
from copy import deepcopy
import json

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool, sum_models, to_classifier

from data_processing import Reader, Checker, DataEncoder, NanProcessor, Shuffler, FeatureAdder, data_loader
from db import db_processor
from api_types import ModelStatuses, ModelTypes
from settings import MODEL_FOLDER, THREAD_COUNT, USE_DETAILED_LOG, USED_RAM_LIMIT, DATASET_BATCH_LENGTH, QUANTIZE

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
        self.bool_columns = ['is_reverse', 'is_main_asset']
        self.float_columns = ['qty',
                              'price',
                              'sum',]        
        self.parameters = {'x_columns': self.x_columns, 
                           'y_columns': self.y_columns, 
                           'str_columns': self.str_columns,
                           'float_columns': self.float_columns,
                           'bool_columns': self.bool_columns,
                           'additional_columns': self.additional_columns}
        
        self.status = ModelStatuses.CREATED
        self.error_text = ''

        self.fitting_start_date: Optional[datetime] = None
        self.fitting_end_date: Optional[datetime] = None

        self.metrics = {}

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

        self.field_models = {}
        self.test_field_models = {}         
        self.data_encoder = None  

        self.strict_acc = {}
        self.test_strict_acc = {}
        self.need_to_encode = True
        self.classes = {}  

        self.metrics_dataset_name = ''
        self.test_metrics_dataset_name = ''        

    async def fit(self, parameters):
        
        logger.info("Fitting")             
        try:
            need_to_initialize = self.status in [ModelStatuses.CREATED, ModelStatuses.ERROR] or parameters.get('refit') == 0
            calculate_metrics = parameters.get('calculate_metrics')
            use_cross_validation = parameters.get('use_cross_validation')

            await self._before_fit(parameters, need_to_initialize, calculate_metrics, use_cross_validation)         
            X_y = await self._read_dataset(parameters)

            train_test_indexes = None
            self.metrics_dataset_name = ''
            self.test_metrics_dataset_name = ''
            if use_cross_validation:
                train_test_indexes = self._get_train_test_indexes(X_y)
                if calculate_metrics:
                    self.metrics_dataset_name = await self._save_dataset_to_temp(X_y.iloc[train_test_indexes[0]])
                    self.test_metrics_dataset_name = await self._save_dataset_to_temp(X_y.iloc[train_test_indexes[1]])  
            else:
                if calculate_metrics:
                    self.metrics_dataset_name = await self._save_dataset_to_temp(X_y)                  

            datasets = await self._transform_dataset(X_y, parameters, need_to_initialize, train_test_indexes, calculate_metrics)

            await self._fit(datasets['train'], parameters, is_first=need_to_initialize)       
                         
            if calculate_metrics:
                await self._calculate_metrics(parameters, need_to_initialize, use_cross_validation)                   

            await self._after_fit(parameters, need_to_initialize=need_to_initialize, use_cross_validation=use_cross_validation)

        except Exception as e:
            await self._on_fitting_error(e)
    
    @abstractmethod
    async def _fit(self, dataset, parameters, is_first=True):
        ...

    @abstractmethod
    async def predict(self,  X, for_metrics=False):
        ...

    def _get_train_test_indexes(self, X_y):
        indexes_len = X_y.shape[0]
        indexes = np.arange(indexes_len)

        test_size=0.2

        train_indexes = indexes[:int(indexes_len*(1-test_size))]
        test_indexes = indexes[int(indexes_len*(1-test_size)):]

        return train_indexes, test_indexes

    async def _get_train_test_datasets(self, X_y, train_indexes, test_indexes, calculate_metrics, use_cross_validation):

        train_dataset = X_y.iloc[train_indexes]
        test_dataset = X_y.iloc[test_indexes]

        to_add = []
        to_delete = []
        for y_col in self.y_columns:
            train_values = set(train_dataset[y_col].unique())
            test_values = set(test_dataset[y_col].unique())

            for val in train_values:
                if val not in test_values:
                    to_add.append(val)

            for val in test_values:
                if val not in train_values:
                    to_delete.append(val)    

            if to_delete:
                test_dataset = test_dataset.loc[~test_dataset[y_col].isin(to_delete)]

            if to_add:
                add_dataset = train_dataset.loc[train_dataset[y_col].isin(to_add)]
                test_dataset = pd.concat([test_dataset, add_dataset])

        return train_dataset, test_dataset    

    async def _get_metric(self, dataset):

        logger.info('Get metrics')       
        metrics = -1
           
        c_x_columns = self.x_columns 
        dataset['check'] = 0

        to_rename = {el: '{}_val'.format(el) for el in self.y_columns}
        dataset = dataset.rename(to_rename, axis=1)

        dataset_pred = await self.predict(dataset, for_metrics=True)

        for y_col in self.y_columns:

            check_column = '{}_check'.format(y_col)
            pred_column = '{}_pred'.format(y_col)
            val_column = '{}_val'.format(y_col)

            dataset[pred_column] = dataset_pred[y_col]
            logger.info('Calculating metrics , field "{}"'.format(y_col)) 

            dataset[check_column] = dataset[val_column] != dataset[pred_column] 

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
            metrics = right/all

        return metrics

    async def _before_fit(self, parameters, need_to_initialize, calculate_metrics, use_cross_validation):

        self.status = ModelStatuses.FITTING
        self.fitting_start_date = datetime.now(UTC)  

        if need_to_initialize:
            self.__init__(self.base_name)
            self.need_to_encode = parameters.get('need_to_encode', True)   
            self._delete_all_models()
        
    async def _after_fit(self, parameters, need_to_initialize, use_cross_validation):
        await self.save()
        if self.metrics_dataset_name:
            await self._delete_dataset_from_temp(self.metrics_dataset_name)
            self.metrics_dataset_name = ''
        if self.test_metrics_dataset_name:
            await self._delete_dataset_from_temp(self.test_metrics_dataset_name)
            self.test_metrics_dataset_name = ''
        self.status = ModelStatuses.READY
        self.fitting_end_date = datetime.now(UTC)   
        logger.info("Fitting. Done")   

    async def _read_dataset(self, parameters):
        
            data_filter = parameters['data_filter']
            if USE_DETAILED_LOG:
                logger.info("Reading data from db")
            reader = Reader()
            X_y = await reader.read(data_filter)

            return X_y
    
    async def _transform_dataset(self, dataset, parameters, need_to_initialize, train_test_indexes=None, calculate_metrics=False):
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")

        pipeline_list = []
        pipeline_list.append(('checker', Checker(self.parameters)))
        pipeline_list.append(('nan_processor', NanProcessor(self.parameters)))        
        pipeline_list.append(('feature_adder', FeatureAdder(self.parameters)))  
        if self.need_to_encode:
            if need_to_initialize:
                self.data_encoder = DataEncoder(self.parameters)  
            self.data_encoder.form_encode_dict = need_to_initialize 
            pipeline_list.append(('data_encoder', self.data_encoder)) 

        pipeline_list.append(('shuffler', Shuffler(self.parameters)))                 

        pipeline = Pipeline(pipeline_list)
        dataset = pipeline.fit_transform(dataset)

        datasets = {}

        if train_test_indexes:
            datasets['train'] = dataset.iloc[train_test_indexes[0]]
            datasets['test'] = dataset.iloc[train_test_indexes[1]]             
        else:
            datasets['train'] = dataset     

        return datasets

    async def _on_fitting_error(self, ex):

        self.status = ModelStatuses.ERROR
        self.error_text = str(ex) 
        if self.metrics_dataset_name:
            await self._delete_dataset_from_temp(self.metrics_dataset_name)
            self.metrics_dataset_name = ''
        if self.test_metrics_dataset_name:
            await self._delete_dataset_from_temp(self.test_metrics_dataset_name)
            self.test_metrics_dataset_name = ''

        await self.save(without_models=True)

        raise ex

    async def _calculate_metrics(self, parameters, need_to_initialize=False, use_cross_validation=False):
        if USE_DETAILED_LOG:
            logger.info("Start calculating metrics")
        dataset = await self._load_dataset_from_temp(self.metrics_dataset_name)
        self.metrics['train'] = await self._get_metric(dataset)
        if use_cross_validation:
            test_dataset = await self._load_dataset_from_temp(self.test_metrics_dataset_name)
            self.metrics['test'] = await self._get_metric(test_dataset)            

        if USE_DETAILED_LOG:
            logger.info("Calculating metrics. Done")     

    @abstractmethod
    def _save_column_model(self, column, item=None):
        ...

    @abstractmethod
    def _load_column_model(self, column, item=None):
        ...

    def _save_parameters(self):
        path_to_model = os.path.join(MODEL_FOLDER, self.uid)
        if not os.path.isdir(path_to_model):
            os.makedirs(path_to_model)

        parameters = deepcopy(self.parameters)
        parameters['uid'] = self.uid
        parameters['model_type'] = self.model_type.value
        parameters['base_name'] = self.base_name
        parameters['metrics'] = self.metrics
        parameters['need_to_encode'] = self.need_to_encode
        parameters['status'] = self.status.value
        parameters['classes'] = {k: [int(el) for el in v] for k, v in self.classes.items()}               
        with open(os.path.join(path_to_model, 'parameters.json'), 'w') as fp:
            json.dump(parameters, fp)

    def _load_parameters(self):
        with open(os.path.join(MODEL_FOLDER, self.uid, 'parameters.json'), 'r') as fp:
            parameters = json.load(fp)

        self.parameters = parameters

        self.base_name = parameters['base_name']    

        self.x_columns = parameters['x_columns']
        self.y_columns = parameters['y_columns']  
        
        self.additional_columns = parameters['additional_columns']

        self.str_columns = parameters['str_columns']
        self.bool_columns = parameters['bool_columns']
        self.float_columns = parameters['float_columns']        

        self.metrics = parameters['metrics']
        self.status = ModelStatuses(parameters['status'])


        self.columns_to_encode = parameters['columns_to_encode']                   

        self.need_to_encode = parameters['need_to_encode']
        self.classes =  {k: [np.int64(el) for el in v] for k, v in parameters['classes'].items()}

    def _save_encoder(self):
        if self.need_to_encode:
            path_to_model = os.path.join(MODEL_FOLDER, self.uid)
            if not os.path.isdir(path_to_model):
                os.makedirs(path_to_model)

            with open(os.path.join(path_to_model, 'encoder.pkl'), 'wb') as fp:
                pickle.dump(self.data_encoder, fp)

    def _load_encoder(self):
        if self.need_to_encode:
            path_to_model = os.path.join(MODEL_FOLDER, self.uid)
            if os.path.exists(path_to_model):
                with open(os.path.join(path_to_model, 'encoder.pkl'), 'rb') as fp:
                    self.data_encoder = pickle.load(fp)

    def _delete_all_models(self):
        path_to_dir = os.path.join(MODEL_FOLDER, self.uid)

        if os.path.exists(path_to_dir):
            for y_col in self.y_columns:
                if os.path.isdir(os.path.join(path_to_dir, y_col)):
                    shutil.rmtree(os.path.join(path_to_dir, y_col))
        path_to_encoder = os.path.join(path_to_dir, 'encoder.pkl')
        if os.path.exists(path_to_encoder):
            os.remove(path_to_encoder)

    async def load(self, uid):
        self.uid = uid

        self._load_parameters()
        
        if self.status != ModelStatuses.ERROR:
            for y_col in self.y_columns:

                if y_col == 'cash_flow_details_code':
                    for item in self.classes[y_col]:
                        self._load_column_model(y_col, item)
                else:
                    self._load_column_model(y_col)

            self._load_encoder()

    async def save(self, without_models=False):
        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)

        path_to_model = os.path.join(MODEL_FOLDER, self.uid)
        if os.path.isdir(path_to_model):
            shutil.rmtree(path_to_model)

        if not without_models:
            for y_col in self.y_columns:
                path_to_col = os.path.join(MODEL_FOLDER, self.uid, y_col)
                if not os.path.isdir(path_to_col):
                    os.makedirs(path_to_col)
                if y_col == 'cash_flow_details_code':
                    for item in self.classes[y_col]:
                        self._save_column_model(y_col, item)
                else:
                    self._save_column_model(y_col)
            self._save_encoder()
        self._save_parameters()

    async def _save_dataset_to_temp(self, dataset):
        collection_name = 'temp_{}'.format(uuid.uuid4())
        await db_processor.insert_many(collection_name, dataset[self.x_columns + self.y_columns].to_dict(orient='records'))
        return collection_name

    async def _load_dataset_from_temp(self, collection_name):
        dataset = await db_processor.find(collection_name)
        dataset = pd.DataFrame(dataset)
        return dataset
    
    async def _delete_dataset_from_temp(self, collection_name):
        await db_processor.delete_many(collection_name)                


class CbCallBack:
    def after_iteration(self, info):
        if USE_DETAILED_LOG:
            if info.metrics.get('validation'):
                logger.info("{}: - loss = {}, test loss = {}".format(info.iteration, info.metrics['learn'][list(info.metrics['learn'].keys())[0]][-1],
                                                                     info.metrics['validation'][list(info.metrics['validation'].keys())[0]][-1]))
            else:
                logger.info("{}: - loss = {}".format(info.iteration, info.metrics['learn'][list(info.metrics['learn'].keys())[0]][-1]))
        return True  


class CatBoostModel(Model):
    model_type = ModelTypes.catboost
    def __init__(self, base_name):
        super().__init__(base_name)

    async def _fit(self, dataset, parameters, is_first=True):
        if USE_DETAILED_LOG:
            logger.info('{} fit'.format('First' if is_first else 'continuous'))  
        pools = dataset
        if is_first:
            self.strict_acc = {}
            self.test_strict_acc = {}

        c_x_columns= self.x_columns.copy() 

        indexes_to_encode = []
        for ind, col in enumerate(self.x_columns):
            if col in self.columns_to_encode:
                indexes_to_encode.append(ind)
        t_indexes_to_encode = indexes_to_encode.copy()            

        for y_col in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start Fitting model. Field = "{}"'.format(y_col))   
            if y_col != 'cash_flow_details_code':

                pool = pools[y_col]
                if isinstance(pool, list):
                    models = []
                                        
                    for ind, c_pool in enumerate(pool):
                        if USE_DETAILED_LOG:
                            logger.info("Model {}".format(ind)) 
                                                   
                        c_model = self._get_cb_model(parameters)
                        c_model.set_params(class_names=self.classes[y_col])
                        c_model.fit(c_pool, callbacks=[CbCallBack()], verbose=False)
                        self._save_cb_model(c_model, y_col, number=ind)
                        del c_model
                        gc.collect()

                    models = self._load_column_models(y_col).values() 
                    if not is_first:
                        models = [self.field_models[y_col]] + models                   
                    model = to_classifier(sum_models(models)) if len(models) > 1 else models[0]
                    self.field_models[y_col] = model 
                    self._save_cb_model(model, y_col)
                    del model
                    for c_model in models:
                        del c_model
                    gc.collect()

                    self._delete_submodels(y_col)

                elif isinstance(pool, Pool):
                    init_model = self.field_models[y_col] if not is_first else None               
                    c_model = self._get_cb_model(parameters) 
                    c_model.fit(pool, callbacks=[CbCallBack()], verbose=False, init_model=init_model)
                    self._save_cb_model(c_model, y_col)
                    del c_model
                    gc.collect()                    
                else:
                    self.strict_acc[y_col] = pool
            else:
                self.field_models[y_col] = {}
                self.strict_acc[y_col] = {}

                c_pools = pools[y_col]

                ind = 0
                for item_col, c_pool in c_pools.items():
                    if USE_DETAILED_LOG:
                        logger.info("Fitting {} - {}".format(ind, item_col))                    
                    if isinstance(c_pool, Pool):
                        init_model = self.field_models[y_col][item_col] if not is_first else None
                        c_model = self._get_cb_model(parameters)
  
                        c_model.fit(c_pool, callbacks=[CbCallBack()], verbose=False, init_model=init_model)

                        self._save_cb_model(c_model, column=y_col, item=item_col)
                        del c_model
                        gc.collect()
                    else:
                        self.strict_acc[y_col][item_col] = c_pool

                    ind += 1
            c_x_columns.append(y_col) 
            indexes_to_encode.append(len(c_x_columns))  

            self._load_all_models()

    async def predict(self, X, for_metrics=False):
        
        if not for_metrics and self.status != ModelStatuses.READY:
            raise ValueError('Model is not ready. Fit it before.')
        
        field_models = self.field_models
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")
        X = pd.DataFrame(X)
        row_numbers = list(X.index)
        X_result = X.copy()

        pipeline_list = []
        pipeline_list.append(('checker', Checker(self.parameters, for_predict=True)))
        pipeline_list.append(('nan_processor', NanProcessor(self.parameters, for_predict=True)))  
        pipeline_list.append(('feature_addder', FeatureAdder(self.parameters, for_predict=True)))  
        if self.need_to_encode:
            pipeline_list.append(('data_encoder', self.data_encoder))                     

        pipeline = Pipeline(pipeline_list)

        for y_col in self.y_columns:
            X[y_col] = ''

        X_y = pipeline.transform(X).copy()

        c_x_columns= self.x_columns
        
        for y_col in self.y_columns:
            if USE_DETAILED_LOG:            
                logger.info('Start predicting. Field = "{}"'.format(y_col))
            if y_col == 'cash_flow_details_code':

                cash_flow_items = list(X_y['cash_flow_item_code'].unique())
                X_y['row_number'] = row_numbers
                X_y_list = []

                for ind, item_col in enumerate(cash_flow_items):
                    if USE_DETAILED_LOG:                    
                        logger.info('Predicting {} - {}'.format(ind, item_col))

                    c_X_y = X_y.loc[X_y['cash_flow_item_code'] == item_col].copy()
                    
                    if (self.strict_acc.get(y_col) is not None 
                        and self.strict_acc[y_col].get(item_col) is not None):
                        c_X_y[y_col] =  self.strict_acc[y_col][item_col]
                    else:
                    
                        X = c_X_y[self.x_columns].to_numpy()

                        c_model = field_models[y_col][str(item_col)]
                        y_pred = c_model.predict(X)
                        c_X_y[y_col] = y_pred.ravel()

                    X_y_list.append(c_X_y)

                t_X_y = pd.concat(X_y_list, axis=0)
                if y_col in X_y.columns:
                    X_y = X_y.drop([y_col], axis=1)

                X_y = X_y.merge(t_X_y[['row_number', y_col]], on=['row_number'], how='left')
                X_y = X_y.set_index(X_y['row_number'])
            else:
                c_y_columns = [y_col]
                if self.strict_acc.get(y_col) is not None:
                    X_y[y_col] =  self.strict_acc[y_col][item_col]
                else:
                    X = X_y[c_x_columns].to_numpy()

                    model = field_models[y_col]
                    y = model.predict(X)
                    X_y[y_col] = y.ravel()
                if USE_DETAILED_LOG:
                    logger.info('Predicting model. Field = "{}". Done'.format(y_col))            

            c_x_columns = c_x_columns + c_y_columns  

        if self.need_to_encode:
            X_y = pipeline.named_steps['data_encoder'].inverse_transform(X_y)  

        for col in self.y_columns:
            X_result[col] = X_y[col]
        
        if for_metrics:
            return X_result
        else:
            return X_result.to_dict(orient='records')       

    def _get_dataset_with_right_classes(self, dataset, x_columns, y_column, model_classes=None, all_dataset=None):

        data_classes = set(dataset[y_column].unique())
        
        if model_classes is None and len(data_classes) <= 1:
            return None
        elif model_classes is not None:
            model_classes = set(model_classes)

            if data_classes == model_classes:
                result = dataset[x_columns+[y_column]]
            else:
                to_delete = []
                to_add = []
                for cl in model_classes:
                    if cl not in data_classes:
                        to_add.append(cl)
                for cl in data_classes:
                    if cl not in model_classes:
                        to_delete.append(cl)

                if to_delete:
                    result = dataset.loc[~dataset[y_column].isin(to_delete)]
                else:
                    result = dataset

                if to_add:
                    to_concat = []
                    for val in to_add:

                        add_dataset = all_dataset.loc[all_dataset[y_column] == val]
                        if len(add_dataset) > 0:
                            if len(add_dataset) >= 50:
                                add_dataset = add_dataset.iloc[:50]
                            to_concat.append(add_dataset)
                        else:
                            none_str = data_loader.get_none_data_row(self.parameters)
                            none_str[y_column] = val
                            to_concat.append(none_str)
                    
                    result = pd.concat([result] + to_concat)
                
                print(to_delete, to_add)

                result = result[x_columns+[y_column]] 
        else:
            result = dataset[x_columns+[y_column]]          

        return result

    def _get_cb_model(self, parameters):
        epochs = parameters.get('epochs', 20)
        depth = parameters.get('depth', 8)
        model = CatBoostClassifier(iterations=epochs, 
                                   learning_rate=0.1, 
                                   depth=depth, 
                                   thread_count=THREAD_COUNT,
                                   used_ram_limit=USED_RAM_LIMIT)
        
        return model

    def _get_data_pools(self, dataset, test_dataset=None):
        pools= {}
        test_pools = None
        use_cross_validation = test_dataset is not None
        c_x_columns = self.x_columns.copy()
        to_delete = []
        cash_flow_items = list(dataset['cash_flow_item_code'].unique())
        for y_col in self.y_columns:
            if y_col != 'cash_flow_details_code':
                value_items = list(dataset[y_col].unique())

                if DATASET_BATCH_LENGTH > 0 and len(value_items) > 1:
                    begin = 0
                    end = DATASET_BATCH_LENGTH
                    pools[y_col] = []              
                    while True:
                        b_dataset = dataset.iloc[begin:min([end, len(dataset)])]
                        c_dataset = self._get_dataset_with_right_classes(b_dataset, c_x_columns, y_col, model_classes=value_items, all_dataset=dataset)
                        c_pool = Pool(c_dataset[c_x_columns], c_dataset[y_col])
                        if QUANTIZE:
                            c_pool.quantize()
                        pools[y_col].append(c_pool)
                        to_delete.append(c_dataset) 
                        to_delete.append(b_dataset)                       
                        if end >= len(dataset):
                            break
                        begin += DATASET_BATCH_LENGTH
                        end += DATASET_BATCH_LENGTH                      
                else:
                    c_dataset = self._get_dataset_with_right_classes(dataset, c_x_columns, y_col)
                    if c_dataset is not None:
                        pools[y_col] = Pool(c_dataset[c_x_columns], c_dataset[y_col])
                        if QUANTIZE:
                            pools[y_col].quantize()
                        to_delete.append(c_dataset)                     
                    else:
                        pools[y_col] = dataset.iloc[0][y_col]

            else:
                pools[y_col] = {}
                for ind, item_col in enumerate(cash_flow_items):
                    c_dataset = dataset.loc[dataset['cash_flow_item_code'] == item_col]
                    value = c_dataset.iloc[0][y_col]
                    c_dataset = self._get_dataset_with_right_classes(c_dataset, c_x_columns, y_col)                    
                    if c_dataset is not None:
                        pools[y_col][item_col] = Pool(c_dataset[c_x_columns], c_dataset[y_col])
                        pools[y_col][item_col].quantize()                        
                        to_delete.append(c_dataset)                        
                    else:
                        pools[y_col][item_col] = value

            c_x_columns.append(y_col)

        to_delete.append(dataset)

        for el in to_delete:
            del el
            el = pd.DataFrame()

        gc.collect()

        return pools, test_pools

    def _save_cb_model(self, model: CatBoostClassifier, column, item=None, number=None):
        path_to_model_folder = os.path.join(MODEL_FOLDER, self.uid, column)
        if item is not None:
            path_to_model = os.path.join(path_to_model_folder, '{}.cbm'.format(item))
        elif number is not None:
            path_to_model = os.path.join(path_to_model_folder, '{}.cbm'.format(str(number)))
        else:
            path_to_model = os.path.join(path_to_model_folder, 'sum.cbm')

        if not os.path.isdir(path_to_model_folder):
            os.makedirs(path_to_model_folder) 
        
        model.save_model(path_to_model)

    def _load_cb_model(self, column, item=None, number=None) -> CatBoostClassifier:
        path_to_model_folder = os.path.join(MODEL_FOLDER, self.uid, column)
        if item is not None:
            path_to_model = os.path.join(path_to_model_folder, '{}.cbm'.format(item))
        elif number is not None:
            path_to_model = os.path.join(path_to_model_folder, '{}.cbm'.format(str(number)))
        else:
            path_to_model = os.path.join(path_to_model_folder, 'sum.cbm')

        if not os.path.isdir(path_to_model_folder):
            os.makedirs(path_to_model_folder) 
        model = CatBoostClassifier()
        model.load_model(path_to_model)
        if item is not None:
            if not self.field_models.get(column):
                self.field_models[column] = {}
            self.field_models[column][str(item)] = model
        else:
            self.field_models[column] = model

    def _load_column_models(self, column, sum_model=False):
        folder = os.path.join(MODEL_FOLDER, self.uid, column)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        filenames = os.listdir(folder)
        models = {}

        for filename in filenames:
            if filename == 'sum.cbm' and not sum_model:
                continue
            if filename != 'sum.cbm' and sum_model:
                continue

            item = filename.split('.')[0]
            model = CatBoostClassifier()
            model.load_model(os.path.join(MODEL_FOLDER, self.uid, column, filename))
            models[item] = model

        return models

    def _load_all_models(self):
        for y_col in self.y_columns:
            models = self._load_column_models(y_col, sum_model = y_col != 'cash_flow_details_code')
            if models:
                if y_col != 'cash_flow_details_code':
                    self.field_models[y_col] = list(models.values())[0]
                else:
                    for item, model in models.items():
                        self.field_models[y_col][item] = model

    def _delete_submodels(self, column):
        path_to_dir = os.path.join(MODEL_FOLDER, self.uid, column)

        filenames = os.listdir(path_to_dir)
        for filename in filenames:
            if filename == 'sum.cbm':
                continue

            os.remove(os.path.join(path_to_dir, filename))

    async def _transform_dataset(self, dataset, parameters, need_to_initialize, train_test_indexes=None, calculate_metrics=False):
        datasets = await super()._transform_dataset(dataset, parameters, need_to_initialize, train_test_indexes, calculate_metrics)
        pools = {}
        self.classes = {}
        if need_to_initialize:
            for y_col in self.y_columns:
                self.classes[y_col] = list(datasets['train'][y_col].unique())
        train_pools, test_pools = self._get_data_pools(datasets['train'], datasets.get('test'))
        del datasets['train']
        datasets['train'] = pd.DataFrame()           
        pools['train'] = train_pools

        if datasets.get('test') is not None:           
            del datasets['test']
            datasets['test'] = pd.DataFrame()            

        gc.collect()

        return pools

    async def _on_fitting_error(self, ex):
        self._delete_all_models()            
        await super()._on_fitting_error(ex)

    def _save_column_model(self, column, item=None):

        if not os.path.isdir(os.path.join(MODEL_FOLDER, self.uid, column)):
            os.makedirs(os.path.join(MODEL_FOLDER, self.uid, column))

        if item is not None:
            if self.strict_acc.get(column) and self.strict_acc[column].get(item):
                value = self.strict_acc[column][item]
                with open(os.path.join(MODEL_FOLDER, self.uid, column, '{}.json'.format(item)), 'w') as fp:
                    json.dump({'value': int(value)}, fp)
            elif self.field_models.get(column) and self.field_models.get(column).get(str(item)):
                self._save_cb_model(self.field_models[column][str(item)], column, item)
        else:
            if self.strict_acc.get(column):
                with open(os.path.join(MODEL_FOLDER, self.uid, column, 'sum.json'), 'w') as fp:
                    value = self.strict_acc[column]
                    json.dump({'value': int(value)}, fp)
            elif self.field_models.get(column):
                self._save_cb_model(self.field_models[column], column)            

    def _load_column_model(self, column, item=None):
        if item  is not None:
            if os.path.exists(os.path.join(MODEL_FOLDER, self.uid, column, '{}.json'.format(item))):
                if not self.strict_acc.get(column):
                    self.strict_acc[column] = {}
                with open(os.path.join(MODEL_FOLDER, self.uid, column, '{}.json'.format(item))) as fp:
                    self.strict_acc[column][item] = np.int64(json.load(fp)['value'])
            elif os.path.exists(os.path.join(MODEL_FOLDER, self.uid, column, '{}.cbm'.format(item))):
                self._load_cb_model(column, item)
        else:
            if os.path.exists(os.path.join(MODEL_FOLDER, self.uid, column, 'sum.json')):
                with open(os.path.join(MODEL_FOLDER, self.uid, column, 'sum.json')) as fp:
                    self.strict_acc[column] = np.int64(json.load(fp)['value'])
            elif os.path.exists(os.path.join(MODEL_FOLDER, self.uid, column, 'sum.cbm')):
                self._load_cb_model(column) 
  

class ModelManager:

    def __init__(self):
        self.models = []

    async def read_models(self):

        models = []
        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)
        
        model_dirs = os.listdir(MODEL_FOLDER)

        for model_dir in model_dirs:
            if not os.path.isdir(os.path.join(MODEL_FOLDER, model_dir)):
                continue

            path_to_parameters = os.path.join(MODEL_FOLDER, model_dir, 'parameters.json')
            if not os.path.exists(path_to_parameters):
                continue

            model = None
            # try:
            with open(path_to_parameters, 'r') as fp:
                parameters = json.load(fp)

            model = self._get_new_model(ModelTypes(parameters['model_type']), parameters['base_name'])
            await model.load(parameters['uid'])

            # except Exception as e:
            #     pass

            if model:
                models.append({'model_type': model.model_type, 'base_name': model.base_name, 'model': model})

        self.models = models

    def add_model(self, model):
        model_list = [el for el in self.models if el['model_type'] == model.model_type and el['base_name'] == model.base_name]

        if not model_list:
            self.models.append({'model_type': model.model_type, 'base_name': model.base_name, 'model': model})

    async def write_model(self, model):
        await model.save()

    def get_model(self, model_type=ModelTypes.rf, base_name=''):

        model_list = [el for el in self.models if el['model_type'] == model_type and el['base_name'] == base_name]
        if model_list:
            model = model_list[0]['model']
        else:
            model = self._get_new_model(model_type, base_name)
        return model
    
    def _get_new_model(self, model_type=ModelTypes.rf, base_name=''):

        sublasses = self._get_all_model_subclasses()
        model_classes = [el for el in sublasses if getattr(el, 'model_type') == model_type]
        if not model_classes:
            raise ValueError('Model type "{}" not allowed'.format(model_type))
        model_class = model_classes[0]
        return model_class(base_name=base_name)
 
    def _get_all_model_subclasses(self, model_class=None):

        if not model_class:
            model_class = Model

        subclasses = model_class.__subclasses__()
        result = subclasses.copy()

        for cl in subclasses:
            sub_subclasses = self._get_all_model_subclasses(cl)
            result.extend(sub_subclasses)

        return result

    async def delete_model(self, model_type=ModelTypes.rf, base_name=''):
        
        c_models = [el for el in self.models if el['model_type'] != model_type and el['base_name'] == base_name]
        if c_models:
            model = c_models[0]
        else: 
            model = self.get_model(model_type, base_name)

        self.models = [el for el in self.models if el['model_type'] != model_type and el['base_name'] != base_name]

        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)

        model_dir = os.path.join(MODEL_FOLDER, model.uid)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)

    async def get_info(self, model_type=ModelTypes.rf, base_name=''):

        model = self.get_model(model_type=model_type, base_name=base_name)

        return {'status': model.status, 
                'error_text': model.error_text, 
                'fitting_start_date': model.fitting_start_date, 
                'fitting_end_date': model.fitting_end_date,
                'metrics': model.metrics}        


model_manager = ModelManager()