from sklearn.ensemble import RandomForestClassifier
from typing import Any
import pandas as pd
from datetime import datetime, timezone
import shutil, os, zipfile
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import logging
import time

from api_types import ModelStatuses
from _to_delete_db_connectors.connector import BaseConnector
from errors import ModuleBaseException
from _to_delete_models.transformers import prepare_to_fit, encode_objects_fit, tramsform_data, transform_to_predict, decode_objects
import joblib, json
from pathlib import Path
from zipfile import ZipFile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

MODEL_GROUP = None
MODEL_ART = {}
MODEL_YEAR = None
MODEL_DET = {}

class Processor:

    def __init__(self, db_connector: BaseConnector):
        self.targets = ['group', 'article_cash_flow', 'details_cash_flow', 'year']
        self.db_connector: BaseConnector = db_connector
        self.status = ModelStatuses.NOTFIT
      
    def _save_model(self, model, target):
        model_name = 'model_' + target
        path = 'saved_models/'
        joblib.dump(model, path + model_name, compress=5)
        
    def _load_model(self, target):
        model_name = 'model_' + target
        path = 'saved_models/'
        loaded_model = joblib.load(path + model_name)
        return loaded_model
    
    def unzip_file(self, loaded_file):
        start_time = time.time()
        os.makedirs('unpacked_files', exist_ok=True)
        os.makedirs('loaded_files', exist_ok=True)
        file_name = Path(loaded_file.filename).stem
        with open(f'loaded_files/{loaded_file.filename}', 'wb') as buffer:
            shutil.copyfileobj(loaded_file.file, buffer)
        logger.info('save file')
        with ZipFile(f'loaded_files/{file_name}.zip', 'r') as zip:
            zip.extractall('unpacked_files')
        logger.info('Unzip DONE------')
        with open(f'unpacked_files/{file_name}.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info('Get data DONE------')
        os.remove(f'unpacked_files/{file_name}.json')
        os.remove(f'loaded_files/{file_name}.zip')
        end_time = time.time()
        print('Unzip time: ', end_time - start_time)
        return data

    def fit(self, data: pd.DataFrame):        
        if os.path.exists('saved_models'):
            self.drop_fitting()
        os.makedirs('saved_models')
            
        if self.status == ModelStatuses.INPROGRESS:
            raise ModuleBaseException('Model is fitting in other process')
        self.db_connector.update_status('Model_info', 'fitting_start_date', datetime.now(timezone.utc))
        self.status = ModelStatuses.INPROGRESS
        self.db_connector.update_status('Model_info', 'Status', 'in_progress')
        df_transform = tramsform_data(data)
        df_encode, target_dict = encode_objects_fit(df_transform)
        for key in target_dict.keys():
            self.db_connector.set_lines(key, [target_dict[key]])
        logger.info("SAVE DICTS---------DONE")
        
        for target in self.targets:
            if target == 'article_cash_flow':
                names_groups= list(df_encode['group'].unique())
                for name in names_groups:
                    df_temp = df_encode[df_encode['group'] == names_groups[name]]
                    x, y = prepare_to_fit(df_temp, 'article_cash_flow')
                    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                    model.fit(x, y)
                    logger.info(f"FIT MODEL {name}---------DONE")
                    model_name = 'article' + str(name)
                    self._save_model(model, model_name)
            elif target == 'details_cash_flow':
                names_art= list(df_encode['article_cash_flow'].unique())
                for name in names_art:
                    df_temp = df_encode[df_encode['article_cash_flow'] == names_art[name]]
                    x, y = prepare_to_fit(df_temp, 'details_cash_flow')
                    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                    model.fit(x, y)
                    logger.info(f"FIT MODEL {name}---------DONE")
                    model_name = 'details' + str(name)
                    self._save_model(model, model_name)
            else:
                x, y = prepare_to_fit(df_encode, target)
                model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                model.fit(x, y)
                logger.info(f"FIT MODEL {target}---------DONE")
                self._save_model(model, target)
                logger.info(f"SAVE MODEL {target}---------DONE")
                accuracy = self.metrics(x, y, model)
                logger.info(f'Accuracy {target} ----- {accuracy}')    
                
        self.status = ModelStatuses.FIT
        self.db_connector.update_status('Model_info', 'Status', 'fit')
        logger.info("--------FIT  DONE--------")
        self.db_connector.update_status('Model_info', 'fitting_end_date', datetime.now(timezone.utc))

                
    def get_info(self) -> dict[str, Any]:
        try:
            value = self.db_connector.get_line('Model_info')
            result = {'status': value['Status'],
                    'fitting_start_date': value['fitting_start_date'],
                    'fitting_end_date': value['fitting_end_date']}
        except TypeError:
            result = {'status': 'not_fit',
                    'fitting_start_date': None,
                    'fitting_end_date': None} 
        finally:
            return result
        
    
    def drop_fitting(self):
        for collection in self.db_connector._db.list_collection_names():
            if collection not in ['data', 'Model_info']:
                self.db_connector._db.drop_collection(collection)
        self.db_connector.update_status('Model_info', 'Status', 'not_fit')
        self.db_connector.update_status('Model_info', 'fitting_start_date', None)
        self.db_connector.update_status('Model_info', 'fitting_end_date', None)
        try:
            shutil.rmtree("saved_models")
        except FileNotFoundError as ex:
            print('Модель не была обучена')
            
              
    def metrics(self, x: pd.DataFrame, y: pd.DataFrame, model):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
        preds = model.predict(x_test)
        accuracy = accuracy_score(y_test, preds)
        return accuracy
    
       
    def predict(self, data: pd.DataFrame):
        start_time = time.time()
        # data_json = data.to_dict(orient="records")
        # print(data_json)
        data.loc[data['price'] == '', 'price'] = 0
        data_result = data.copy()    
        data_to_predict = transform_to_predict(self.db_connector, data)
        # pred_json = data_to_predict.to_dict(orient="records")
        # print(pred_json)

        group_time = time.time()
        preds = MODEL_GROUP.predict(data_to_predict.drop(columns=['document', 'group', 'article_cash_flow', 'details_cash_flow', 'year'], axis=1))
        # print(preds)
        data_to_predict['group'] = preds
        decode_preds = decode_objects(self.db_connector, 'group', preds)
        data_result['group'] = decode_preds
        logger.info("group--------DONE")
        group_end_time = time.time()     
        
        for row in range(len(data)):        
            mod_name = 'model_article' + str(data_to_predict['group'].iloc[row])
            preds = MODEL_ART[mod_name].predict(data_to_predict[row:row+1].drop(columns=['document', 'article_cash_flow', 'details_cash_flow', 'year'], axis=1))
            data_to_predict['article_cash_flow'][row] = preds[0]
            logger.info("article_cash_flow--------DONE")
        decode_preds = decode_objects(self.db_connector, 'article_cash_flow', data_to_predict['article_cash_flow'])
        data_result['article_cash_flow'] = decode_preds 

        for row in range(len(data)):        
            mod_name = 'model_details' + str(data_to_predict['article_cash_flow'].iloc[row])
            preds = MODEL_DET[mod_name].predict(data_to_predict[row:row+1].drop(columns=['document', 'details_cash_flow', 'year'], axis=1))
            data_to_predict['details_cash_flow'][row] = preds[0]
            logger.info("details_cash_flow--------DONE")
        decode_preds = decode_objects(self.db_connector, 'details_cash_flow', data_to_predict['details_cash_flow'])
        data_result['details_cash_flow'] = decode_preds
            
        year_time = time.time()           
        preds = MODEL_YEAR.predict(data_to_predict.drop(columns=['document', 'year'], axis=1))
        decode_preds = decode_objects(self.db_connector, 'year', preds)
        data_to_predict['year'] = preds
        data_result['year'] = decode_preds
        logger.info("year--------DONE")
        end_time = time.time()
        print('year time: ', end_time - year_time)
        print('group time: ', group_end_time - group_time)
        result_json = data_result.to_dict(orient="records")
        end_time = time.time()
        print('Predict time: ', end_time - start_time)
        # print(result_json)
        return result_json
    
    def set_global(self):
        global MODEL_ART, MODEL_YEAR, MODEL_DET, MODEL_GROUP
        MODEL_GROUP = self._load_model('group')
        for name in os.listdir('saved_models'):
            if 'article' in name:
                MODEL_ART[name] = self._load_model(name[6:])
        for name in os.listdir('saved_models'):
            if 'details' in name:
                MODEL_DET[name] = self._load_model(name[6:])
        MODEL_YEAR = self._load_model('year')
        logger.info("--------Global models DONE--------")
