from sklearn.ensemble import RandomForestClassifier
from typing import Any
import pandas as pd
from datetime import datetime, timezone
import shutil, os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from api_types import ModelStatuses
from db_connectors.connector import BaseConnector
from errors import ModuleBaseException
from models.transformers import prepare_to_fit, encode_objects_fit, tramsform_data, transform_to_predict, decode_objects
import joblib


class Processor:

    def __init__(self, db_connector: BaseConnector):
        self.targets = ['article_cash_flow', 'details_cash_flow', 'year']
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
        print("SAVE DICTS---------DONE")
        for target in self.targets:
            if target == 'details_cash_flow':
                names_art= list(df_encode['article_cash_flow'].unique())
                for name in names_art:
                    df_temp = df_encode[df_encode['article_cash_flow'] == names_art[name]]
                    x, y = prepare_to_fit(df_temp, 'details_cash_flow')
                    model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                    model.fit(x, y)
                    print(f"FIT MODEL {name}---------DONE")
                    self._save_model(model, str(name))
                    print(f"SAVE MODEL {name}---------DONE")
            else:
                x, y = prepare_to_fit(df_encode, target)
                model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=5)
                model.fit(x, y)
                print(f"FIT MODEL {target}---------DONE")
                self._save_model(model, target)
                print(f"SAVE MODEL {target}---------DONE")
                accuracy = self.metrics(x, y, model)
                print(f'Accuracy {target} ----- {accuracy}')
                
        self.status = ModelStatuses.FIT
        self.db_connector.update_status('Model_info', 'Status', 'fit')
        print("--------ALL DONE--------")
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
        shutil.rmtree("saved_models")
              
    def metrics(self, x: pd.DataFrame, y: pd.DataFrame, model):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
        preds = model.predict(x_test)
        accuracy = accuracy_score(y_test, preds)
        return accuracy
    
    def predict(self, data: pd.DataFrame):
        data_result = data.copy()    
        for row in range(len(data)):
            data_to_predict = transform_to_predict(self.db_connector, data[row:row+1])
            model = self._load_model('article_cash_flow')
            preds = model.predict(data_to_predict.drop(columns=['document', 'article_cash_flow', 'details_cash_flow', 'year'], axis=1))
            decode_preds = decode_objects(self.db_connector, 'article_cash_flow', preds)
            data_to_predict['article_cash_flow'] = preds[0]
            data_result['article_cash_flow'][row:row+1] = decode_preds[0]
            print("article_cash_flow--------DONE")
                
            mod_name = str(data_to_predict['article_cash_flow'].iloc[0])
            model = self._load_model(mod_name[0])
            preds = model.predict(data_to_predict.drop(columns=['document', 'details_cash_flow', 'year'], axis=1))
            data_to_predict['details_cash_flow'] = preds[0]
            decode_preds = decode_objects(self.db_connector, 'details_cash_flow', preds)
            data_result['details_cash_flow'][row:row+1] = decode_preds[0]
            print("details_cash_flow--------DONE")
                
            model = self._load_model('year')
            preds = model.predict(data_to_predict.drop(columns=['document', 'year'], axis=1))
            decode_preds = decode_objects(self.db_connector, 'year', preds)
            data_to_predict['year'] = preds[0]
            data_result['year'][row:row+1] = decode_preds[0]
            print("year--------DONE")
        
        result_json = data_result.to_dict(orient="records")
        return result_json
