import os

DB_HOST = os.getenv('DB_HOST', default='localhost')
DB_PORT = int(os.getenv('DB_PORT', default='27017'))
DB_USER = os.getenv('DB_USER', default='')
DB_PASSWORD = os.getenv('DB_PASSWORD', default='')
DB_AUTH_SOURCE = os.getenv('DB_AUTH_SOURCE', default='')


