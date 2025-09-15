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
from catboost import CatBoostClassifier, sum_models

from data_processing import Reader, Checker, DataEncoder, NanProcessor, Shuffler, FeatureAdder, data_loader
from db import db_processor
from api_types import ModelStatuses, ModelTypes
from settings import MODEL_FOLDER, THREAD_COUNT, USE_DETAILED_LOG

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

    @abstractmethod
    async def fit(self, parameters):
        ...
    
    @abstractmethod
    async def predict(self, x):
        ...

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

    async def _get_metrics(self, datasets):
        # predictions to calculate metrics
        logger.info('Get metrics')       
        metrics = {}
           
        for name, dataset in datasets.items():
            c_x_columns = self.x_columns 
            dataset['check'] = 0

            to_rename = {el: '{}_val'.format(el) for el in self.y_columns}
            dataset = dataset.rename(to_rename, axis=1)
            use_test_models = name in ['test', 'train']
            dataset_pred = await self.predict(dataset, for_metrics=True, use_test_models=use_test_models)

            for y_col in self.y_columns:

                check_column = '{}_check'.format(y_col)
                pred_column = '{}_pred'.format(y_col)
                val_column = '{}_val'.format(y_col)

                dataset[pred_column] = dataset_pred[y_col]
                logger.info('Calculating metrics dataset "{}", field "{}"'.format(name, y_col)) 

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
            metrics['acuracy_{}'.format(name)] = right/all

        return metrics

    async def write_to_db(self):
        if not os.path.isdir(MODEL_FOLDER):
            os.makedirs(MODEL_FOLDER)  

        model_file = os.path.join(MODEL_FOLDER, '{}.mdl'.format(self.uid)) 
        with open(model_file, 'wb') as fp:
            model_bin = pickle.dumps(self)
            model_bin_compressed = zlib.compress(model_bin, level=9)
            fp.write(model_bin_compressed) 


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
        self.test_field_models = {} 

        self.strict_acc = {}
        self.test_strict_acc = {}     
                        
        self.data_encoder = None       

    async def fit(self, parameters):
        
        try:

            need_to_initialize = self.status in [ModelStatuses.CREATED, ModelStatuses.ERROR]
            calculate_metrics = parameters.get('calculate_metrics')
            use_cross_validation = parameters.get('use_cross_validation')

            self.status = ModelStatuses.FITTING
            self.fitting_start_date = datetime.now(UTC)

            if need_to_initialize:
                self.__init__(self.base_name)

            data_filter = parameters['data_filter']
            logger.info("Reading data from db")
            X_y = await Reader().read(data_filter)

            train_indexes, test_indexes = self._get_train_test_indexes(X_y)

            logger.info("Transforming and checking data")
            # pipeline = Pipeline([
            #                     ('checker', Checker(self.parameters)),
            #                     ('nan_processor', NanProcessor(self.parameters)),                            
            #                     ('ferature_adder', FeatureAdder(self.parameters)),
            #                     ('shuffler', Shuffler(self.parameters)),
            #                     ])

            pipeline = Pipeline([
                                ('checker', Checker(self.parameters)),
                                ('nan_processor', NanProcessor(self.parameters)),
                                ('ferature_adder', FeatureAdder(self.parameters)),                            
                                ('data_encoder', DataEncoder(self.parameters)),
                                ('shuffler', Shuffler(self.parameters)),
                                ])        

            X_y = pipeline.fit_transform(X_y, [])

            if use_cross_validation:
                X_y_train, X_y_test = self._get_train_test_datasets(X_y, train_indexes, test_indexes)
            else:
                X_y_train, X_y_test = None, None

            c_x_columns= self.x_columns

            indexes_to_encode = []
            for ind, col in enumerate(self.x_columns):
                if col in self.columns_to_encode:
                    indexes_to_encode.append(ind)
            t_indexes_to_encode = indexes_to_encode.copy()
            
            for y_col in self.y_columns:
                
                logger.info('Start Fitting model. Field = "{}"'.format(y_col))

                if y_col == 'cash_flow_details_code':
                    self.field_models[y_col] = {}
                    if use_cross_validation:
                        self.test_field_models[y_col] = {}
                    cash_flow_items = list(X_y['cash_flow_item_code'].unique())

                    for ind, item_col in enumerate(cash_flow_items):
                        logger.info("Fitting {} - {}".format(ind, item_col))

                        c_X_y = X_y.loc[X_y['cash_flow_item_code'] == item_col].copy()
                    
                        if len(c_X_y[y_col].unique()) == 1:
                            self.strict_acc[item_col] = c_X_y[y_col].unique()[0]
                        else:
                            X = c_X_y[self.x_columns].to_numpy()
                            y = c_X_y[[y_col]].to_numpy().ravel()                             
                            
                            c_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                            c_model.fit(X, y)
                            self.field_models[y_col][item_col] = c_model

                        if use_cross_validation:
                            c_X_y_train = X_y_train.loc[X_y_train['cash_flow_item_code'] == item_col].copy()


                            if len(c_X_y_train) == 0:
                                self.test_strict_acc[item_col] = ''                                
                            elif len(c_X_y_train[y_col].unique()) == 1:
                                self.test_strict_acc[item_col] = c_X_y_train[y_col].unique()[0]
                            else:
                                X_train = c_X_y_train[self.x_columns].to_numpy()
                                y_train = c_X_y_train[[y_col]].to_numpy().ravel()                             

                                c_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                                c_model.fit(X_train, y_train)
                                self.test_field_models[y_col][item_col] = c_model

                else:
                    c_y_columns = [y_col]

                    X, y = X_y[c_x_columns].to_numpy(), X_y[c_y_columns].to_numpy().ravel()
                    if  use_cross_validation:
                        X_train, y_train = X_y_train[c_x_columns].to_numpy(), X_y_train[c_y_columns].to_numpy().ravel()

                    if need_to_initialize:
                        model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                        if use_cross_validation:
                            model_test = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                    else:
                        model = self.field_models[y_col]
                        if use_cross_validation:
                            model_test = self.test_field_models[y_col]

                    model.fit(X, y)
                    self.field_models[y_col] = model
                    
                    if use_cross_validation:
                        model_test.fit(X_train, y_train)
                        self.test_field_models[y_col] = model_test   

                    logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

                indexes_to_encode.append(len(c_x_columns))
                c_x_columns = c_x_columns + c_y_columns

            self.data_encoder = pipeline.named_steps['data_encoder']

            if calculate_metrics:
                datasets = {'all': X_y}
                
                if use_cross_validation:
                    datasets['train'] = X_y_train 
                    datasets['test'] = X_y_test 

                logger.info("Start calculating metrics")
                self.metrics = await self._get_metrics(datasets) 
                logger.info("Calculating metrics. Done")                     

            self.status = ModelStatuses.READY
            self.fitting_end_date = datetime.now(UTC)            
        except Exception as e:
            self.status = ModelStatuses.ERROR
            self.error_text = str(e)

            await self.write_to_db()
            raise e

    # async def predict(self, X, need_to_encode=True):
        
    #     if self.status != ModelStatuses.READY:
    #         raise ValueError('Model is not ready. Fit it before.')

    #     logger.info("Transforming and checking data")
    #     X = pd.DataFrame(X)
    #     X_result = X.copy()

    #     pipeline_list = [
    #                     ('checker', Checker(self.parameters, for_predict=True)),
    #                     ('nan_processor', NanProcessor(self.parameters, for_predict=True)),
    #                     ('ferature_adder', FeatureAdder(self.parameters))                            
    #                     ]
        
    #     if need_to_encode:
    #         pipeline_list.append(('data_encoder', self.data_encoder))

    #     pipeline = Pipeline(pipeline_list)

    #     X_y = pipeline.transform(X).copy()
    #     c_x_columns= self.x_columns
        
    #     for y_col in self.y_columns:
    #         c_y_columns = [y_col]

    #         X = X_y[c_x_columns].to_numpy()
    #         logger.info('Start predicting. Field = "{}"'.format(y_col))

    #         model = self.field_models[y_col]
    #         y = model.predict(X)

    #         X_y[y_col] = y

    #         logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

    #         c_x_columns = c_x_columns + c_y_columns  

    #     if need_to_encode:
    #         X_y = pipeline.named_steps['data_encoder'].inverse_transform(X_y)

    #     for col in self.y_columns:
    #         X_result[col] = X_y[col]

    #     return X_result.to_dict(orient='records') 
    # 
    async def predict(self, X, for_metrics=False, use_test_models=False):
        
        if not for_metrics and self.status != ModelStatuses.READY:
            raise ValueError('Model is not ready. Fit it before.')
        
        field_models = self.test_field_models if use_test_models else self.field_models

        logger.info("Transforming and checking data")
        X = pd.DataFrame(X)
        X_result = X.copy()

        if not for_metrics:
            pipeline_list = [
                            ('checker', Checker(self.parameters, for_predict=True)),
                            ('nan_processor', NanProcessor(self.parameters, for_predict=True)),
                            ('feature_addder', FeatureAdder(self.parameters, for_predict=True)),
                            ('data_encoder',  self.data_encoder),                                                                                 
                            ]        

            pipeline = Pipeline(pipeline_list)

            X_y = pipeline.transform(X).copy()
        else:
            X_y = X
        c_x_columns= self.x_columns
        
        for y_col in self.y_columns:
            logger.info('Start predicting. Field = "{}"'.format(y_col))
            if y_col == 'cash_flow_details_code':

                cash_flow_items = list(X_y['cash_flow_item_code'].unique())
                X_y['row_number'] = X_y.index
                X_y_list = []

                for ind, item_col in enumerate(cash_flow_items):
                    logger.info('Predicting {} - {}'.format(ind, item_col))
                    c_X_y = X_y.loc[X_y['cash_flow_item_code'] == item_col].copy()
                    
                    if item_col not in field_models[y_col]:
                        if item_col not in self.strict_acc:
                            c_X_y[y_col] = ''
                        else:
                            c_X_y[y_col] = self.strict_acc[item_col]
                    else:
                        X = c_X_y[self.x_columns].to_numpy()

                        c_model = field_models[y_col][item_col]
                        y_pred = c_model.predict(X)
                        c_X_y[y_col] = y_pred.ravel()

                    X_y_list.append(c_X_y)

                t_X_y = pd.concat(X_y_list, axis=0)
                if y_col in X_y.columns:
                    X_y = X_y.drop([y_col], axis=1)

                X_y = X_y.merge(t_X_y[['row_number', y_col]], on=['row_number'], how='left')

            else:
                c_y_columns = [y_col]

                X = X_y[c_x_columns].to_numpy()

                model = field_models[y_col]
                y = model.predict(X)

                X_y[y_col] = y.ravel()

                logger.info('Predicting model. Field = "{}". Done'.format(y_col))            

            c_x_columns = c_x_columns + c_y_columns  

        for col in self.y_columns:
            X_result[col] = X_y[col]
        
        if for_metrics:
            return X_result
        else:
            return X_result.to_dict(orient='records')                      


class CbCallBack:
    def after_iteration(self, info):
        if USE_DETAILED_LOG:
            logger.info("{}: - loss {}".format(info.iteration, info.metrics['learn'][list(info.metrics['learn'].keys())[0]][-1]))
        return True  


class CatBoostModel(Model):
    model_type = ModelTypes.catboost
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

        self.field_models = {}
        self.test_field_models = {}         
        self.data_encoder = None  

        self.strict_acc = {}
        self.test_strict_acc = {}     

    async def fit(self, parameters):
        
        files_to_delete = []   
        logger.info("Fitting")             
        try:
            epochs = parameters.get('epochs', 20)
            depth = parameters.get('depth', 8)   

            need_to_initialize = self.status in [ModelStatuses.CREATED, ModelStatuses.ERROR] or parameters.get('refit') == 0
            calculate_metrics = parameters.get('calculate_metrics')
            use_cross_validation = parameters.get('use_cross_validation')

            self.status = ModelStatuses.FITTING
            self.fitting_start_date = datetime.now(UTC)

            if need_to_initialize:
                self.__init__(self.base_name)

            data_filter = parameters['data_filter']
            if USE_DETAILED_LOG:
                logger.info("Reading data from db")
            X_y = await Reader().read(data_filter)

            train_indexes, test_indexes = self._get_train_test_indexes(X_y)
            if USE_DETAILED_LOG:
                logger.info("Transforming and checking data")
            pipeline = Pipeline([
                                ('checker', Checker(self.parameters)),
                                ('nan_processor', NanProcessor(self.parameters)),                            
                                ('ferature_adder', FeatureAdder(self.parameters)),
                                ('shuffler', Shuffler(self.parameters)),
                                ])

            X_y = pipeline.fit_transform(X_y, [])

            if use_cross_validation:
                X_y_train, X_y_test = self._get_train_test_datasets(X_y, train_indexes, test_indexes)
            else:
                X_y_train, X_y_test = None, None

            if need_to_initialize:
                await self._fit_first(X_y, 
                                        epochs=epochs, 
                                        depth=depth,
                                        use_cross_validation=use_cross_validation,
                                        X_y_train=X_y_train,
                                        X_y_test=X_y_test)
            else:
                await self._fit_continous(X_y, 
                                        epochs=epochs, 
                                        depth=depth,                                        
                                        use_cross_validation=use_cross_validation,
                                        X_y_train=X_y_train,
                                        X_y_test=X_y_test,
                                        files_to_delete=files_to_delete)                
            if calculate_metrics:
                datasets = {'all': X_y}
                
                if use_cross_validation:
                    datasets['train'] = X_y_train 
                    datasets['test'] = X_y_test 
                if USE_DETAILED_LOG:
                    logger.info("Start calculating metrics")
                self.metrics = await self._get_metrics(datasets) 
                if USE_DETAILED_LOG:
                    logger.info("Calculating metrics. Done")                     

            self.status = ModelStatuses.READY
            self.fitting_end_date = datetime.now(UTC)   
            logger.info("Fitting. Done")                         
        except Exception as e:

            self.status = ModelStatuses.ERROR
            self.error_text = str(e)
            
            await self.write_to_db()
            raise e

    async def _fit_first(self, X_y, epochs, depth, use_cross_validation, X_y_train, X_y_test):

        if USE_DETAILED_LOG:
            logger.info('First fit')   
        self.strict_acc = {}
        self.test_strict_acc = {}
        c_x_columns= self.x_columns

        indexes_to_encode = []
        for ind, col in enumerate(self.x_columns):
            if col in self.columns_to_encode:
                indexes_to_encode.append(ind)
        t_indexes_to_encode = indexes_to_encode.copy()
        
        for y_col in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start Fitting model. Field = "{}"'.format(y_col))         

            if y_col == 'cash_flow_details_code':

                self.field_models[y_col] = {}
                self.strict_acc[y_col] = {}
                if use_cross_validation:
                    self.test_field_models[y_col] = {}
                    self.test_strict_acc[y_col] = {}

                cash_flow_items = list(X_y['cash_flow_item_code'].unique())

                for ind, item_col in enumerate(cash_flow_items):
                    if USE_DETAILED_LOG:
                        logger.info("Fitting {} - {}".format(ind, item_col))

                    c_X_y = X_y.loc[X_y['cash_flow_item_code'] == item_col].copy()
                
                    X, y = self._get_x_y_with_right_classes(c_X_y, self.x_columns, y_col)  
                    if X is None:
                        self.strict_acc[y_col][item_col] = c_X_y.iloc[0][y_col]
                    else:                                           
                        c_model = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)

                        c_model.fit(X, y, cat_features=t_indexes_to_encode, callbacks=[CbCallBack()], verbose=False)
                            
                        self.field_models[y_col][item_col] = c_model

                    if use_cross_validation:
                        c_X_y_train = X_y_train.loc[X_y_train['cash_flow_item_code'] == item_col].copy()
                        X_train, y_train = self._get_x_y_with_right_classes(c_X_y_train, self.x_columns, y_col)
                        if X_train is None:
                            self.test_strict_acc[y_col][item_col] = c_X_y_train.iloc[0][y_col]
                        else:                            
                            c_model_test = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)                                 

                            c_model_test.fit(X_train, y_train, cat_features=t_indexes_to_encode, callbacks=[CbCallBack()], verbose=False)
                                
                            self.test_field_models[y_col][item_col] = c_model_test

            else:
                c_y_columns = [y_col]

                X, y = self._get_x_y_with_right_classes(X_y, c_x_columns, y_col) 
                if X is None:
                    self.strict_acc[y_col] = X_y.iloc[0][y_col]
                else:
                    model = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)
                    model.fit(X, y, cat_features=indexes_to_encode, callbacks=[CbCallBack()], verbose=False)
                    self.field_models[y_col] = model  

                if  use_cross_validation:
                    X_train, y_train = self._get_x_y_with_right_classes(X_y_train, c_x_columns, y_col)
                    if X_train is None:
                        self.strict_acc[y_col] = X_y_train.iloc[0][y_col]
                    else:
                        model_test = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)                    

                        model_test.fit(X_train, y_train, cat_features=indexes_to_encode, callbacks=[CbCallBack()], verbose=False)
                        self.test_field_models[y_col] = model_test   
                if USE_DETAILED_LOG:
                    logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

            indexes_to_encode.append(len(c_x_columns))
            c_x_columns = c_x_columns + c_y_columns

    async def _fit_continous(self, X_y, epochs, depth, use_cross_validation, X_y_train, X_y_test, files_to_delete):
        if USE_DETAILED_LOG:
            logger.info('Continous fit')   
        c_x_columns= self.x_columns

        indexes_to_encode = []
        for ind, col in enumerate(self.x_columns):
            if col in self.columns_to_encode:
                indexes_to_encode.append(ind)
        t_indexes_to_encode = indexes_to_encode.copy()

        for y_col in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start Fitting model. Field = "{}"'.format(y_col))         

            if y_col == 'cash_flow_details_code':

                cash_flow_items = list(self.field_models['cash_flow_item_code'].classes_)

                for ind, item_col in enumerate(cash_flow_items):
                    if USE_DETAILED_LOG:                    
                        logger.info("Fitting {} - {}".format(ind, item_col))

                    c_X_y = X_y.loc[X_y['cash_flow_item_code'] == item_col].copy()
                
                    c_model = self.field_models[y_col][item_col]
                    c_model_new = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)

                    X, y = self._get_x_y_with_right_classes(c_X_y, self.x_columns, y_col, c_model.classes_)      
                    
                    if X is None:
                        self.strict_acc[item_col] = c_X_y.iloc[0][y_col].value
                    else:
                        c_model_new.fit(X, y, cat_features=t_indexes_to_encode, callbacks=[CbCallBack()], verbose=False, 
                                    init_model=c_model)
                            
                        self.field_models[y_col][item_col] = c_model_new

                    if use_cross_validation:

                        c_X_y_train = X_y_train.loc[X_y['cash_flow_item_code'] == item_col].copy()
                    
                        c_model_test = self.test_field_models[y_col][item_col]
                        c_model_test_new = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)                      
                        
                        X_train, y_train = self._get_x_y_with_right_classes(c_X_y_train, self.x_columns, y_col, self.test_field_models[y_col][item_col].classes_)                                                                      

                        c_model_test_new.fit(X_train, y_train, cat_features=t_indexes_to_encode, callbacks=[CbCallBack()], verbose=False, 
                                         init_model=c_model_test)
                            
                        self.test_field_models[y_col][item_col] = c_model_test_new                        

            else:
                c_y_columns = [y_col]
                if  self.field_models.get(y_col) is not None:

                    X, y = self._get_x_y_with_right_classes(X_y, c_x_columns, y_col, self.field_models[y_col].classes_) 

                    if X is not None or self.strict_acc.get(y_col) is not None:
                        pass
                    else:
                        model = self.field_models[y_col]                              
                        model_new = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)                    
                        model_new.fit(X, y, cat_features=indexes_to_encode, callbacks=[CbCallBack()], verbose=False, 
                            init_model=model)
                        self.field_models[y_col] = model_new
                
                if  use_cross_validation:
                    if  self.test_field_models.get(y_col) is not None:

                        X_train, y_train = self._get_x_y_with_right_classes(X_y_train, c_x_columns, y_col, self.test_field_models[y_col].classes_)  
                        
                        if X_train is not None or self.test_strict_acc.get(y_col) is not None:
                            pass
                        else:
                            model_test = self.test_field_models[y_col] 
                            model_test_new = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)
                
        
                            model_test_new.fit(X_train, y_train, cat_features=indexes_to_encode, callbacks=[CbCallBack()], verbose=False, 
                                        init_model=model_test)
                            self.test_field_models[y_col] = model_test_new   
                if USE_DETAILED_LOG:
                    logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

            indexes_to_encode.append(len(c_x_columns))
            c_x_columns = c_x_columns + c_y_columns

    async def predict(self, X, for_metrics=False, use_test_models=False):
        
        if not for_metrics and self.status != ModelStatuses.READY:
            raise ValueError('Model is not ready. Fit it before.')
        
        field_models = self.test_field_models if use_test_models else self.field_models
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")
        X = pd.DataFrame(X)
        row_numbers = list(X.index)
        X_result = X.copy()

        if not for_metrics:
            pipeline_list = [
                            ('checker', Checker(self.parameters, for_predict=True)),
                            ('nan_processor', NanProcessor(self.parameters, for_predict=True)),
                            ('feature_addder', FeatureAdder(self.parameters, for_predict=True)),                                                    
                            ]

            pipeline = Pipeline(pipeline_list)

            X_y = pipeline.transform(X).copy()
        else:
            X_y = X.copy()
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

                        c_model = field_models[y_col][item_col]
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

        for col in self.y_columns:
            X_result[col] = X_y[col]
        
        if for_metrics:
            return X_result
        else:
            return X_result.to_dict(orient='records')       

    def _get_x_y_with_right_classes(self, X_y, x_columns, y_column, model_classes=None):

        data_classes = set(X_y[y_column].unique())
        
        if model_classes is None and len(data_classes) <= 1:
            # X_y = data_loader.get_none_data_row(self.parameters)
            # none_str = data_loader.get_none_data_row(self.parameters)
            # none_str[y_column] = 'None_2'
            # X_y = pd.concat([X_y, none_str])
            return None, None
        elif model_classes is not None:
            
            model_classes = set(model_classes)

            if data_classes == model_classes:
                X, y = X_y[x_columns].to_numpy(), X_y[[y_column]].to_numpy()
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
                    X_y = X_y.loc[~X_y[y_column].isin(to_delete)].copy()

                if to_add:
                    to_concat = []
                    for val in to_add:
                        none_str = data_loader.get_none_data_row(self.parameters)
                        none_str[y_column] = val
                        to_concat.append(none_str)
                    
                    X_y = pd.concat([X_y] + to_concat)
                
                print(to_delete, to_add)
        
        X, y = X_y[x_columns].to_numpy(), X_y[[y_column]].to_numpy()                

        return X, y


class CatBoostEncModel(CatBoostModel):
    model_type = ModelTypes.catboostenc

    async def fit(self, parameters):
        
        files_to_delete = []   
        logger.info("Fitting")             
        try:
            epochs = parameters.get('epochs', 20)
            depth = parameters.get('depth', 8)

            need_to_initialize = self.status in [ModelStatuses.CREATED, ModelStatuses.ERROR] or parameters.get('refit') == 0
            calculate_metrics = parameters.get('calculate_metrics')
            use_cross_validation = parameters.get('use_cross_validation')

            self.status = ModelStatuses.FITTING
            self.fitting_start_date = datetime.now(UTC)

            if need_to_initialize:
                self.__init__(self.base_name)

            data_filter = parameters['data_filter']
            if USE_DETAILED_LOG:
                logger.info("Reading data from db")
            X_y = await Reader().read(data_filter)

            train_indexes, test_indexes = self._get_train_test_indexes(X_y)
            if USE_DETAILED_LOG:
                logger.info("Transforming and checking data")
            pipeline = Pipeline([
                                ('checker', Checker(self.parameters)),
                                ('nan_processor', NanProcessor(self.parameters)),                            
                                ('ferature_adder', FeatureAdder(self.parameters)),
                                ('data_encoder', DataEncoder(self.parameters)),                                
                                ('shuffler', Shuffler(self.parameters)),
                                ])

            X_y = pipeline.fit_transform(X_y, [])

            if use_cross_validation:
                X_y_train, X_y_test = self._get_train_test_datasets(X_y, train_indexes, test_indexes)
            else:
                X_y_train, X_y_test = None, None

            if need_to_initialize:
                await self._fit_first(X_y, 
                                        epochs=epochs,
                                        depth=depth,                                         
                                        use_cross_validation=use_cross_validation,
                                        X_y_train=X_y_train,
                                        X_y_test=X_y_test)
            else:
                await self._fit_continous(X_y, 
                                        epochs=epochs, 
                                        depth=depth,
                                        use_cross_validation=use_cross_validation,
                                        X_y_train=X_y_train,
                                        X_y_test=X_y_test,
                                        files_to_delete=files_to_delete)
                
            self.data_encoder = pipeline.named_steps['data_encoder']

            if calculate_metrics:
                datasets = {'all': X_y}
                
                if use_cross_validation:
                    datasets['train'] = X_y_train 
                    datasets['test'] = X_y_test 
                if USE_DETAILED_LOG:
                    logger.info("Start calculating metrics")
                self.metrics = await self._get_metrics(datasets) 
                if USE_DETAILED_LOG:
                    logger.info("Calculating metrics. Done")                     

            self.status = ModelStatuses.READY
            self.fitting_end_date = datetime.now(UTC)   
            logger.info("Fitting. Done")                         
        except Exception as e:

            self.status = ModelStatuses.ERROR
            self.error_text = str(e)
            
            await self.write_to_db()
            raise e

    async def _fit_first(self, X_y, epochs, depth, use_cross_validation, X_y_train, X_y_test):

        if USE_DETAILED_LOG:
            logger.info('First fit')   
        self.strict_acc = {}
        self.test_strict_acc = {}
        c_x_columns= self.x_columns

        indexes_to_encode = []
        for ind, col in enumerate(self.x_columns):
            if col in self.columns_to_encode:
                indexes_to_encode.append(ind)
        t_indexes_to_encode = indexes_to_encode.copy()
        
        for y_col in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start Fitting model. Field = "{}"'.format(y_col))         

            if y_col == 'cash_flow_details_code':

                self.field_models[y_col] = {}
                self.strict_acc[y_col] = {}
                if use_cross_validation:
                    self.test_field_models[y_col] = {}
                    self.test_strict_acc[y_col] = {}

                cash_flow_items = list(X_y['cash_flow_item_code'].unique())

                for ind, item_col in enumerate(cash_flow_items):
                    if USE_DETAILED_LOG:
                        logger.info("Fitting {} - {}".format(ind, item_col))

                    c_X_y = X_y.loc[X_y['cash_flow_item_code'] == item_col].copy()
                
                    X, y = self._get_x_y_with_right_classes(c_X_y, self.x_columns, y_col)  
                    if X is None:
                        self.strict_acc[y_col][item_col] = c_X_y.iloc[0][y_col]
                    else:                                           
                        c_model = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)

                        c_model.fit(X, y, callbacks=[CbCallBack()], verbose=False)
                            
                        self.field_models[y_col][item_col] = c_model

                    if use_cross_validation:
                        c_X_y_train = X_y_train.loc[X_y_train['cash_flow_item_code'] == item_col].copy()
                        X_train, y_train = self._get_x_y_with_right_classes(c_X_y_train, self.x_columns, y_col)
                        if X_train is None:
                            self.test_strict_acc[y_col][item_col] = c_X_y_train.iloc[0][y_col]
                        else:                            
                            c_model_test = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)                                 

                            c_model_test.fit(X_train, y_train, cat_features=t_indexes_to_encode, callbacks=[CbCallBack()], verbose=False)
                                
                            self.test_field_models[y_col][item_col] = c_model_test

            else:
                c_y_columns = [y_col]

                X, y = self._get_x_y_with_right_classes(X_y, c_x_columns, y_col) 
                if X is None:
                    self.strict_acc[y_col] = X_y.iloc[0][y_col]
                else:
                    model = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)
                    model.fit(X, y, callbacks=[CbCallBack()], verbose=False)
                    self.field_models[y_col] = model  

                if  use_cross_validation:
                    X_train, y_train = self._get_x_y_with_right_classes(X_y_train, c_x_columns, y_col)
                    if X_train is None:
                        self.strict_acc[y_col] = X_y_train.iloc[0][y_col]
                    else:
                        model_test = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)                    

                        model_test.fit(X_train, y_train, callbacks=[CbCallBack()], verbose=False)
                        self.test_field_models[y_col] = model_test   
                if USE_DETAILED_LOG:
                    logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

            indexes_to_encode.append(len(c_x_columns))
            c_x_columns = c_x_columns + c_y_columns

    async def _fit_continous(self, X_y, epochs, depth, use_cross_validation, X_y_train, X_y_test, files_to_delete):
        if USE_DETAILED_LOG:
            logger.info('Continous fit')   
        c_x_columns= self.x_columns

        indexes_to_encode = []
        for ind, col in enumerate(self.x_columns):
            if col in self.columns_to_encode:
                indexes_to_encode.append(ind)
        t_indexes_to_encode = indexes_to_encode.copy()

        for y_col in self.y_columns:
            if USE_DETAILED_LOG:
                logger.info('Start Fitting model. Field = "{}"'.format(y_col))         

            if y_col == 'cash_flow_details_code':

                cash_flow_items = list(self.field_models['cash_flow_item_code'].classes_)

                for ind, item_col in enumerate(cash_flow_items):
                    if USE_DETAILED_LOG:                    
                        logger.info("Fitting {} - {}".format(ind, item_col))

                    c_X_y = X_y.loc[X_y['cash_flow_item_code'] == item_col].copy()
                
                    c_model = self.field_models[y_col][item_col]
                    c_model_new = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)

                    X, y = self._get_x_y_with_right_classes(c_X_y, self.x_columns, y_col, c_model.classes_)      
                    
                    if X is None:
                        self.strict_acc[item_col] = c_X_y.iloc[0][y_col].value
                    else:
                        c_model_new.fit(X, y, callbacks=[CbCallBack()], verbose=False, 
                                    init_model=c_model)
                            
                        self.field_models[y_col][item_col] = c_model_new

                    if use_cross_validation:

                        c_X_y_train = X_y_train.loc[X_y['cash_flow_item_code'] == item_col].copy()
                    
                        c_model_test = self.test_field_models[y_col][item_col]
                        c_model_test_new = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)                      
                        
                        X_train, y_train = self._get_x_y_with_right_classes(c_X_y_train, self.x_columns, y_col, self.test_field_models[y_col][item_col].classes_)                                                                      

                        c_model_test_new.fit(X_train, y_train, callbacks=[CbCallBack()], verbose=False, 
                                         init_model=c_model_test)
                            
                        self.test_field_models[y_col][item_col] = c_model_test_new                        

            else:
                c_y_columns = [y_col]
                if  self.field_models.get(y_col) is not None:

                    X, y = self._get_x_y_with_right_classes(X_y, c_x_columns, y_col, self.field_models[y_col].classes_) 

                    if X is not None or self.strict_acc.get(y_col) is not None:
                        pass
                    else:
                        model = self.field_models[y_col]                              
                        model_new = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)                    
                        model_new.fit(X, y, callbacks=[CbCallBack()], verbose=False, 
                            init_model=model)
                        self.field_models[y_col] = model_new
                
                if  use_cross_validation:
                    if  self.test_field_models.get(y_col) is not None:

                        X_train, y_train = self._get_x_y_with_right_classes(X_y_train, c_x_columns, y_col, self.test_field_models[y_col].classes_)  
                        
                        if X_train is not None or self.test_strict_acc.get(y_col) is not None:
                            pass
                        else:
                            model_test = self.test_field_models[y_col] 
                            model_test_new = CatBoostClassifier(iterations=epochs, learning_rate=0.1, depth=depth, thread_count=THREAD_COUNT)
                
        
                            model_test_new.fit(X_train, y_train, callbacks=[CbCallBack()], verbose=False, 
                                        init_model=model_test)
                            self.test_field_models[y_col] = model_test_new   
                if USE_DETAILED_LOG:
                    logger.info('Fitting model. Field = "{}". Done'.format(y_col))            

            indexes_to_encode.append(len(c_x_columns))
            c_x_columns = c_x_columns + c_y_columns

    async def predict(self, X, for_metrics=False, use_test_models=False):
        
        if not for_metrics and self.status != ModelStatuses.READY:
            raise ValueError('Model is not ready. Fit it before.')
        
        field_models = self.test_field_models if use_test_models else self.field_models
        if USE_DETAILED_LOG:
            logger.info("Transforming and checking data")
        X = pd.DataFrame(X)
        row_numbers = list(X.index)
        X_result = X.copy()

        if not for_metrics:
            pipeline_list = [
                            ('checker', Checker(self.parameters, for_predict=True)),
                            ('nan_processor', NanProcessor(self.parameters, for_predict=True)),
                            ('feature_addder', FeatureAdder(self.parameters, for_predict=True)),
                            ('data_encoder', self.data_encoder),                                                                                
                            ]

            pipeline = Pipeline(pipeline_list)

            X_y = pipeline.transform(X).copy()
        else:
            X_y = X.copy()
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

                        c_model = field_models[y_col][item_col]
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

        if not for_metrics:
            X_y = pipeline.named_steps['data_encoder'].inverse_transform(X_y)  

        for col in self.y_columns:
            X_result[col] = X_y[col]
        
        if for_metrics:
            return X_result
        else:
            return X_result.to_dict(orient='records')


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

    def add_model(self, model):
        model_list = [el for el in self.models if el['model_type'] == model.model_type and el['base_name'] == model.base_name]

        if not model_list:
            self.models.append({'model_type': model.model_type, 'base_name': model.base_name, 'model': model})

    async def write_model(self, model):
        await model.write_to_db()

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