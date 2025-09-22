from contextlib import asynccontextmanager
from typing import Optional
import asyncio
from fastapi import FastAPI, APIRouter, Header, HTTPException, BackgroundTasks, File, UploadFile, Depends, Query, Path, Body
from fastapi.exceptions import HTTPException
from cachetools import TTLCache
import aiohttp
import traceback

from settings import VERSION, DB_URL, TEST_MODE, AUTH_SERVICE_URL
from fastapi.responses import HTMLResponse # FileResponse

import logging
import uuid

from tasks import task_manager
from data_processing import data_loader
from api_types import TaskResponse, ModelTypes, ModelInfo, DataRow, ModelStatuses, StatusResponse
from db import db_processor
from models import model_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

auth_cache = TTLCache(maxsize=1000, ttl=300)


async def check_token(token: str = Header()) -> bool:
    """Проверка токена авторизации с кэшем и защитой от сбоев."""

    if TEST_MODE:
        return True

    if token in auth_cache:
        return True

    timeout = aiohttp.ClientTimeout(total=10)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{AUTH_SERVICE_URL}/check_token", # type: ignore
                json={"token": token}
            ) as response:
                if response.status == 200:
                    auth_cache[token] = True
                    return True
                elif response.status == 404:
                    raise HTTPException(status_code=401, detail="Invalid token")
                elif response.status == 403:
                    raise HTTPException(status_code=401, detail="Token expired")
                else:
                    logger.error(f"Auth service returned unexpected status {response.status}")
                    raise HTTPException(status_code=500, detail="Authentication service error")

    except asyncio.TimeoutError:
        logger.error("Auth request timed out")
        raise HTTPException(status_code=503, detail="Authorization timeout")

    except aiohttp.ClientError as e:
        logger.warning(f"Auth service connection error: {str(e)}")
        raise HTTPException(status_code=503, detail="Authorization service unavailable")

    except Exception as e:
        logger.exception("Unexpected error in check_token")
        raise HTTPException(status_code=500, detail="Internal auth error")


async def get_token_from_header(token: Optional[str] = Header(None, alias="token")):
    return token


@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Starting DB connection")
        app.db = db_processor
        logger.info(DB_URL)        
        await app.db.connect(url=DB_URL)
        await app.db.delete_temp_collections()
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
        app.db.close()
        logger.info("Database connection closed")

    except Exception as e:
        logger.error(f"Database shutdown error: {e}")
        raise
   
    logger.info("Shutting down prediction service...")    


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
        model_manager.add_model(model)
        if model.status == ModelStatuses.FITTING:
            raise ValueError('Current model is already fitting')
        
        parameters = task.parameters
        if not parameters:
            parameters = {}

        if not 'data_filter' in parameters:
            parameters['data_filter'] = {}
        
        data_filter = {'base_name': task.base_name} if task.base_name != 'all_bases' else None
        if data_filter:
            parameters['data_filter'].update(data_filter)
        
        await model.fit(parameters)

        logger.info("Start writing model to db")
        try:
            await model_manager.write_model(model)
        except Exception as e:
            model.status = ModelStatuses.ERROR
            raise e
        logger.info("Writing model to db. Done")     

        # Update status
        logger.info(f"[{task_id}] fitting task completed")
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
        token: str = Depends(get_token_from_header),
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
async def delete_data(base_name: str = Query(default=''),
                      token: str = Depends(get_token_from_header),
                      authenticated: bool = Depends(check_token),) -> str:
    db_filter = None

    if base_name:
        db_filter = {'base_name': base_name}
    await data_loader.delete_data(db_filter)
    return 'Data has been deleted'


@app.post('/fit')
async def fit(background_tasks: BackgroundTasks,
              token: str = Depends(get_token_from_header),
              authenticated: bool = Depends(check_token),
              base_name: str = Query(default=''),
              model_type: ModelTypes = Query(default=ModelTypes.rf),
              parameters: dict = Body(),) -> TaskResponse:

    logger.info(f"Start fitting model")

    task_id = str(uuid.uuid4())
    task = await task_manager.create_task(task_id)

    try:
        if not base_name:
            base_name = 'all_bases'
        # Update task
        await task_manager.update_task(
            task_id,
            type='FIT',
            status="PREPARE_FITTING",
            upload_progress=100,
            model_type=model_type.value,
            base_name=base_name,
            parameters=parameters)

        # Start background task
        background_tasks.add_task(process_fitting_model, task_id)

        return TaskResponse(task_id=task_id, message="Task processing started")
    
    except Exception as e:
        logger.error(f"Error in fitting model: {e}")

        await task_manager.update_task(task_id, status="ERROR", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/delete_model')
async def delete_model(base_name: str = Query(default=''),
                       model_type: ModelTypes = Query(default=ModelTypes.rf),
                       token: str = Depends(get_token_from_header),
                       authenticated: bool = Depends(check_token)) -> str:
    try:
        if not base_name:
            base_name = 'all_bases'        
        await model_manager.delete_model(model_type=model_type, base_name=base_name)
        return 'Model has been deleted'
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        raise HTTPException(status_code=500, detail=str(e))   


@app.get('/get_model_info')
async def get_model_info(base_name: str = Query(default=''),
                         model_type: ModelTypes = Query(default=ModelTypes.rf),
                         token: str = Depends(get_token_from_header),
                         authenticated: bool = Depends(check_token)) -> ModelInfo:
    try:
        if not base_name:
            base_name = 'all_bases'        
        result = await model_manager.get_info(model_type=model_type.value, base_name=base_name)
        return ModelInfo.model_validate(result)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))      


@app.post('/predict')
async def predict(X: list[DataRow], 
                  base_name: str = Query(default=''),
                  model_type: ModelTypes = Query(default=ModelTypes.rf),
                  token: str = Depends(get_token_from_header),
                  authenticated: bool = Depends(check_token)) -> list[DataRow]:
    try:    
        if not base_name:
            base_name = 'all_bases'         
        X_list = []
        for row in X:
            X_list.append(row.model_dump())
        model = model_manager.get_model(model_type, base_name)
        result = []
        X_y_list = await model.predict(X_list)
        for row in X_y_list:
            result.append(DataRow.model_validate(row))
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error predicting: {e}")
        raise HTTPException(status_code=500, detail=str(e))           

    return result


@app.get('/get_task_status')
async def get_task_status(task_id: str = Query(),
                          token: str = Depends(get_token_from_header),
                          authenticated: bool = Depends(check_token)) -> StatusResponse:
    status = await task_manager.get_task_status(task_id)

    return status