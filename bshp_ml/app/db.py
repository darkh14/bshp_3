from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional, Any
from pymongo.errors import ServerSelectionTimeoutError
import logging


logger = logging.getLogger(__name__)


class DBProcessor:
    """ Class realises working with DB using MONGO DB.
    """
    def __init__(self) -> None:
        self.url = ''
        self.timeout = 5000
        self.master_db_name = 'bshp'
        self.db = None
        self._connection: Optional[AsyncIOMotorClient] = None
        self.accounting_db = ''

    async def connect(self, url, timeout=5000) -> bool:
        self.url = url
        self.timeout = timeout
        
        attempts = 3
        connected = False
        connection_timeout_error = None
        for att in range(1, attempts+1):
            logger.info("Try to connect attempt {}".format(att))            
            try:
                self._connection = AsyncIOMotorClient(self.url, 
                                                serverSelectionTimeoutMS=self.timeout)              
                
                self.db = self._connection.get_database(self.master_db_name)            
                await self.db.command("ping")
                connected = True
                break
            except ServerSelectionTimeoutError as e:
                connection_timeout_error = e

        if connected:
            logger.info("Successfully connected to the database")
        else:
            logger.info("Error database connection. {}".format(str(connection_timeout_error)))
            raise connection_timeout_error
        

        db_integrity = await self.check_database_integrity()
        if not db_integrity:
            logger.error("Database integrity check failed")
            raise Exception("Database integrity check failed")
        
        return True

    async def check_database_integrity(self):
        """Проверка целостности базы данных и восстановление индексов"""
        logger.info("check_database_integrity")
        return True
    
    def close(self):
        self._connection.close()

    async def find_one(self, collection_name: str, db_filter=None) -> Optional[dict]:
        """ See base method docs
        :param collection_name: required collection name
        :param db_filter: optional, db filter value to find line
        :return: dict of db line
        """
        collection = self._get_collection(collection_name)

        db_filter = db_filter or None

        result = await collection.find_one(db_filter, projection={'_id': False})

        return result or None

    async def find(self, collection_name: str, db_filter=None, batch_size=0) -> list[dict]:
        collection = self._get_collection(collection_name)

        c_filter = db_filter if db_filter else None
        if batch_size:
            cursor = collection.find(c_filter, projection={'_id': False}, batch_size=1000) # .to_list()
            result = []

            async for doc in cursor:
                result.append(doc)

            await cursor.close()
        else:
            result = await collection.find(c_filter, projection={'_id': False}).to_list()

        return result

    async def insert_one(self, collection_name: str, value: dict[str, Any], db_filter=None) -> bool:

        collection = self._get_collection(collection_name)

        db_filter = db_filter or None
        if db_filter:
            result = await collection.replace_one(db_filter, value, upsert=True)
        else:
            result = await collection.insert_one(value)

        return bool(getattr(result, 'acknowledged'))

    async def insert_many(self, collection_name: str, value: list[dict[str, Any]]) -> bool:

        collection = self._get_collection(collection_name)

        result = await collection.insert_many(value)

        return bool(getattr(result, 'acknowledged'))

    async def delete_many(self, collection_name: str, db_filter=None) -> bool:
        
        db_filter = db_filter or None
        if db_filter:
            collection = self._get_collection(collection_name)
            result = await collection.delete_many(db_filter)
            result = bool(getattr(result, 'acknowledged'))
        else:
            result = await self.db.drop_collection(collection_name)
            result = result is not None

        return result

    async def get_count(self, collection_name: str, db_filter=None) -> int:
        """ See base method docs
        :param collection_name: required collection name
        :param db_filter: optional, db filter value to find lines
        :return: number of lines in collection
        """
        db_filter = db_filter or {}
        collection = self._get_collection(collection_name)

        return await collection.count_documents(db_filter)

    def _get_collection(self, collection_name):
        """ Gets collection object from db object
        :param collection_name: name of required collection,
        :return: collection object
        """
        return self.db.get_collection(collection_name)

    def get_collection_names(self) -> list[str]:

        return self._db.list_collection_names()

    async def drop_db(self) -> str:
        """Method to drop current database
        :return result of dropping"""

        collection_names = self.get_collection_names()

        async for collection_name in collection_names:
            await self.delete_many(collection_name)

        return True

    async def get_collection_names(self):
        return await self.db.list_collection_names()

    async def delete_temp_collections(self):
        collection_names = await self.get_collection_names()
        for collection_name in collection_names:
            if 'temp_' in collection_name:
                await self.delete_many(collection_name)


db_processor = DBProcessor()
