from pydantic import BaseModel, field_validator
from enum import Enum
from datetime import datetime

from typing import Optional, Dict


class TaskData(BaseModel):
    task_id: str
    type: str = 'FIT'
    status: str = "CREATED"
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    base_name: Optional[str] = None
    replace: Optional[bool] = None
    model_type: Optional[str] = None
    parameters: Optional[dict] = None

    # Внутренние поля
    file_path: Optional[str] = None


class TaskResponse(BaseModel):
    task_id: str
    message: str


class StatusResponse(BaseModel):
    status: str
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    description: Optional[str] = None


class ProcessingTaskResponse(BaseModel):
    task_id: str
    type: str
    base_name: str
    status: str    
    

class DataRow(BaseModel):
    """
    Loading, input or output data row
    """
    number: str 
    date: datetime
    is_reverse: bool
    moving_type: str
    kind: str
    company_inn: str
    company_kpp: str
    base_document_number: str
    base_document_date: datetime
    base_document_kind: str
    base_document_operation_type: str
    contractor_name: str
    contractor_inn: str
    contractor_kpp: str
    contractor_kind: str
    article_name: str
    article_code: str
    is_main_asset: bool
    analytic: str
    analytic2: str
    analytic3: str
    article_document_number: str
    article_document_date: datetime    
    article_parent: str
    article_group: str
    article_kind: str
    row_number: int    
    article_row_number: int
    store: str
    department: str
    company_account_number: str
    contractor_account_number: str
    qty: float
    price: float
    sum: float
    cash_flow_item_code: str
    year: str
    cash_flow_details_code: str

    @field_validator('date', mode='before')
    def check_date(cls, value):
        if isinstance(value, str):
            result = datetime.strptime(value, r'%d.%m.%Y %H:%M:%S')
        else: 
            result = value

        return result
    
    @field_validator('base_document_date', mode='before')
    def check_base_document(cls, value):
        if isinstance(value, str):
            result = datetime.strptime(value, r'%d.%m.%Y %H:%M:%S')
        else: 
            result = value

        return result  

    
    @field_validator('article_document_date', mode='before')
    def check_article_document_date(cls, value):
        if not value:
            return datetime(1, 1, 1)
        elif isinstance(value, str):
            result = datetime.strptime(value, r'%d.%m.%Y %H:%M:%S')
        else: 
            result = value

        return result               
    

class ModelStatuses(Enum):
    CREATED = 'CREATED'
    FITTING = 'FITTING'    
    READY = 'READY'
    ERROR = 'ERROR'


class ModelInfo(BaseModel):
    status: ModelStatuses
    error_text: str
    fitting_start_date: Optional[datetime]
    fitting_end_date: Optional[datetime]
    metrics: Optional[Dict[str, float]]


class ModelTypes(str, Enum):
    rf = 'rf'
    catboost = 'catboost'