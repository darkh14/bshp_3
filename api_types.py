from pydantic import BaseModel
from enum import Enum
from datetime import datetime

from typing import Optional


class DataRow(BaseModel):
    """
    Loading, input or output data row
    """
    base_name: str
    document: str
    article_cash_flow: str
    details_cash_flow: str
    qty: float
    price: float
    sum: float
    year: str
    unit_of_count: str
    is_service: bool
    moving_type: str
    base_article: str
    operation_type: str
    date: str
    payment: str
    reverse: bool
    type_of_customer: str
    type_of_contract: str
    account: str
    sub_account: str
    calculation_account: str
    calculation_account_turnover: str
    calculation_account_total: str
    account_kredit: str
    account_debet: str
    account_debet_turnover: str
    account_debet_total: str
    name_of_noomenclature: str
    type_of_noomenclature: str
    view_of_noomenclature: str
    noomenclature_unit: str
    group_of_noomenclature: str
    name_of_noomenclature_sub: str
    type_of_noomenclature_sub: str
    view_of_noomenclature_sub: str
    noomenclature_unit_sub: str
    group_of_noomenclature_sub: str
    group: str
    

class ModelStatuses(Enum):
    NOTFIT = 'not_fit'
    INPROGRESS = 'in_progress'
    FIT = 'fit'
    ERROR = 'error'
    NEEDFIT = 'need_to_fit'


class ModelInfo(BaseModel):
    status: ModelStatuses
    fitting_start_date: Optional[datetime]
    fitting_end_date: Optional[datetime]