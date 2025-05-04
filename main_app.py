import fastapi
import uvicorn
from typing import Any
import pandas as pd
from pandas import DataFrame

from models.processor import Processor
from db_connectors.connector import MongoConnector
from api_types import DataRow, ModelInfo, ModelStatuses
from version import VERSION

app = fastapi.FastAPI(title="BSHP App",  
                      description="Application for AI cash flow parameters predictions!",
                      version=VERSION)


@app.get('/')
async def main_page():
    """
    Root method returns html ok description
    @return: HTML response with ok micro html
    """
    return fastapi.responses.HTMLResponse('<h2>BSHP module</h2> <br> <h3>Connection established</h3>')


@app.get('/health')
def health() -> str:
    return 'service is working'


@app.post('/save_data')
def save_data(data: list[DataRow]) -> str:
    db_connector = MongoConnector("BSHP")
    data_save = [row.model_dump() for row in data]
    db_connector.set_lines('data', data_save)
    db_connector.update_status('Model_info', 'Status', 'need_to_fit')
    return 'data saved'


@app.post('/predict')
def predict(data: list[DataRow]) -> list[DataRow]:
    db_connector = MongoConnector("BSHP")
    processor = Processor(db_connector)
    data_to_predict = pd.DataFrame([row.model_dump() for row in data])
    return processor.predict(data_to_predict)


@app.get('/get_info')
def get_info() -> dict:
    db_connector = MongoConnector("BSHP")
    processor = Processor(db_connector)
    return processor.get_info


@app.get('/drop_fitting')
def drop_fitting() -> str:
    db_connector = MongoConnector("BSHP")
    processor = Processor(db_connector)
    processor.drop_fitting()    
    return 'fitting is dropped'


@app.get('/fit')
def fit(background_tasks: fastapi.BackgroundTasks) -> str:
    db_connector = MongoConnector("BSHP")
    data_fit = pd.DataFrame(db_connector.get_lines('data'))
    background_tasks.add_task(fit_background_new, data_fit)
    return 'model fitting is started'


@app.get('/delete_data')
def delete_data(base_name: str):
    db_connector = MongoConnector("BSHP")
    db_connector.drop_db(base_name)
    return 'Datas has been deleted'


@app.get('/delete_all_data')
def delete_all_data():
    db_connector = MongoConnector("BSHP")
    db_connector.drop_all_db()
    return 'Datas has been deleted'


def fit_background_new(data: DataFrame) -> bool:
    db_connector = MongoConnector("BSHP")
    processor = Processor(db_connector)
    processor.fit(data)
    return True


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8090, log_level="info")