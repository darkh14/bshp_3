from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, APIRouter, Header, HTTPException, BackgroundTasks, File, UploadFile, Depends, Query, Path, Body

# from models.processor import Processor
# from db_connectors.connector import MongoConnector


from settings import VERSION, DB_URL
from fastapi.responses import HTMLResponse # FileResponse

import logging
import uuid

from tasks import task_manager
from data_processing import data_loader
from api_types import TaskResponse, ModelTypes, ModelInfo, DataRow
from db import db_processor
from models import model_manager



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)


async def check_token(token: str = Header()) -> bool:
    """Проверка токена авторизации с кэшем и защитой от сбоев."""
    return True
    # if config.TEST_MODE: # type: ignore
    #     return True

    # if token in auth_cache:
    #     return True

    # timeout = aiohttp.ClientTimeout(total=10)

    # try:
    #     async with aiohttp.ClientSession(timeout=timeout) as session:
    #         async with session.post(
    #             f"{config.AUTH_SERVICE_URL}/check-token", # type: ignore
    #             json={"token": token}
    #         ) as response:
    #             if response.status == 200:
    #                 auth_cache[token] = True
    #                 return True
    #             elif response.status == 404:
    #                 raise HTTPException(status_code=401, detail="Invalid token")
    #             elif response.status == 403:
    #                 raise HTTPException(status_code=401, detail="Token expired")
    #             else:
    #                 logger.error(f"Auth service returned unexpected status {response.status}")
    #                 raise HTTPException(status_code=500, detail="Authentication service error")

    # except asyncio.TimeoutError:
    #     logger.error("Auth request timed out")
    #     raise HTTPException(status_code=503, detail="Authorization timeout")

    # except aiohttp.ClientError as e:
    #     logger.warning(f"Auth service connection error: {str(e)}")
    #     raise HTTPException(status_code=503, detail="Authorization service unavailable")

    # except Exception as e:
    #     logger.exception("Unexpected error in check_token")
    #     raise HTTPException(status_code=500, detail="Internal auth error")
    
    # """
    # Извлекает токен из заголовка запроса.

    # Args:
    #     token: Токен из заголовка X-API-Token

    # Returns:
    #     str: Токен авторизации
    # """
    # return token


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Starting DB connection")
        app.db = db_processor
        await app.db.connect(url=DB_URL)
        logger.info("DB connection done")

    except Exception as e:
        logger.error(f"DB startup error: {e}")
        raise 

    try:
        logger.info("Starting models initialize")
        await model_manager.read_models()
        logger.info("Models initializing done")   
    except Exception as e:
        logger.error(f"Models initializing error: {e}")
        raise         

    yield 

    try:
        """Закрытие подключения к базе данных при остановке"""
        app.db.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Database shutdown error: {e}")
        raise
   
    logger.info("Shutting down transcription service...")    


app = FastAPI(title="BSHP App",  
                      description="Application for AI cash flow parameters predictions!",
                      version=VERSION,
                      lifespan=lifespan)


@app.get('/')
async def main_page():
    """
    Root method returns html ok description
    @return: HTML response with ok micro html
    """
    return HTMLResponse('<h2>BSHP module</h2> <br> <h3>Connection established</h3>')


async def process_uploading_task(task_id: str):
    """Background task for uploading data from file."""

    logger.info(f"[{task_id}] process_uploading_task started")

    try:
        task = await task_manager.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return

        # Execute loading
        result = await data_loader.upload_data_from_file(
            task,
        )

        logger.info(f"[{task_id}] uploading task completed")

        # Update status
        await task_manager.update_task(task_id, status="READY", progress=100)

        logger.info(f"[{task_id}] Task marked READY")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(f"[{task_id}] Error in data loading task: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))


async def process_fitting_model(task_id: str):
    """Background task for fitting model."""

    logger.info(f"[{task_id}] process_fitting_task started")

    try:
        task = await task_manager.get_task(task_id)
        if not task:
            logger.error(f"Task not found: {task_id}")
            return

        # Execute loading
        model = model_manager.get_model(task.model_type, task.base_name)
        data_filter = {'base_name': task.base_name} if task.base_name else None
        await model.fit({'data_filter': data_filter})

        logger.info("Start writing model to db")
        await model_manager.write_model(model)
        logger.info("Writing model to db. Done")

        logger.info(f"[{task_id}] fitting task completed")

        # Update status
        await task_manager.update_task(task_id, status="READY", progress=100)

        logger.info(f"[{task_id}] Task marked READY")

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        logger.exception(f"[{task_id}] Error in model fitting task: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))


@app.get('/health')
def health() -> str:
    return 'service is working'


@app.get('/version')
def version() -> str:
    return VERSION


@app.post('/save_data')
async def save_data(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        base_name: str = Query(),
        authenticated: bool = Depends(check_token),
        replace: bool = Query(default=False)) -> TaskResponse:
    
    logger.info(f"Starting uploading from file: {file.filename}")

    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)

    try:
        # Save file
        content = await file.read()
        file_path = await task_manager.save_upload_file(task_id, file.filename, content) # type: ignore

        # Update task
        await task_manager.update_task(
            task_id,
            type='UPLOAD',
            status="UPLOADING_FILE",
            upload_progress=100,
            file_path=str(file_path),
            base_name=base_name,
            replace=replace)

        # Start background task
        background_tasks.add_task(process_uploading_task, task_id)

        return TaskResponse(task_id=task_id, message="Task processing started")
    
    except Exception as e:
        logger.error(f"Error in uploading task {task_id}: {e}")

        if "File name too long" in str(e):
            await task_manager.update_task(task_id, status="ERROR", error="Имя файла слишком длинное")
            raise HTTPException(
                status_code=500,
                detail="Имя файла слишком длинное. Сократите имя файла и попробуйте загрузить ещё раз"
            )
        else:
            await task_manager.update_task(task_id, status="ERROR", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


@app.get('/delete_data')
async def delete_data(base_name: str) -> str:
    db_filter = None
    if base_name:
        db_filter = {'base_name': base_name}
    await data_loader.delete_data(db_filter)
    return 'Data has been deleted'


@app.get('/fit')
async def fit(background_tasks: BackgroundTasks,
              base_name: str = Query(default=''),
              model_type: ModelTypes = Query(default='rf')) -> TaskResponse:

    logger.info(f"Starting fitting model")

    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)

    try:

        # Update task
        await task_manager.update_task(
            task_id,
            type='FIT',
            status="PREPARE_FITTING",
            upload_progress=100,
            model_type=model_type.value,
            base_name=base_name)

        # Start background task
        background_tasks.add_task(process_fitting_model, task_id)

        return TaskResponse(task_id=task_id, message="Task processing started")
    
    except Exception as e:
        logger.error(f"Error in fitting model: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/delete_model')
async def delete_model(base_name: str = Query(default=''),
              model_type: ModelTypes = Query(default='rf')) -> str:
    try:
        await model_manager.delete_model(model_type=model_type.value, base_name=base_name)
        return 'Model has been deleted'
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))   

@app.get('/get_model_info')
async def get_model_info(base_name: str = Query(default=''),
              model_type: ModelTypes = Query(default='rf')) -> ModelInfo:
    try:
        result = await model_manager.get_info(model_type=model_type.value, base_name=base_name)
        return ModelInfo.model_validate(result)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))      

@app.post('/predict')
async def predict(X: list[DataRow], 
                  base_name: str = Query(default=''),
                  model_type: ModelTypes = Query(default='rf')) -> list[DataRow]:
    
    X_list = []
    for row in X:
        X_list.append(row.model_dump())
    model = model_manager.get_model(model_type, base_name)
    result = []
    X_y_list = await model.predict(X_list)
    for row in X_y_list:
        result.append(DataRow.model_validate(row))

    return result


# @app.get('/get_info')
# def get_info() -> dict:
#     db_connector = MongoConnector("BSHP")
#     processor = Processor(db_connector)
#     return processor.get_info()


# @app.get('/drop_fitting')
# def drop_fitting() -> str:
#     db_connector = MongoConnector("BSHP")
#     processor = Processor(db_connector)
#     processor.drop_fitting()    
#     return 'fitting is dropped'








# @app.get('/delete_all_data')
# def delete_all_data():
#     db_connector = MongoConnector("BSHP")
#     db_connector.drop_all_db()
#     return 'Datas has been deleted'


# def fit_background_new(data: DataFrame) -> bool:
#     db_connector = MongoConnector("BSHP")
#     processor = Processor(db_connector)
#     processor.fit(data)
#     processor.set_global()
#     return True
