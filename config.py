'''
WorkR
Copyright (C) 2023

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import os

class Config(object):
    WORKER_NAME = 'celery@worker'+str(os.environ.get('WORKER_NUMBER'))+'@'+os.environ.get('WORKER_NAME')
    QUEUE = os.environ.get('QUEUE')
    MONKEY_PATCH = os.environ.get('MONKEY_PATCH')
    INITIAL_SETUP = False

    # SQLAlchemy Config
    SECRET_KEY = os.environ.get('SECRET_KEY')
    SQLALCHEMY_DATABASE_SERVER =  os.environ.get('DATABASE_SERVER')
    SQLALCHEMY_DATABASE_NAME =  os.environ.get('DATABASE_NAME')
    SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_SERVER+"/"+SQLALCHEMY_DATABASE_NAME
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    DETECTOR_THRESHOLDS = {
        'MDv4': 0.8,
        'MDv5a': 0.2,
        'MDv5b': 0.1,
        'error': 1.0,
        'golden': 0.9,
        'user': 0
    }
