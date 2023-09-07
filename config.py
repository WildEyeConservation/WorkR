'''
Copyright 2023

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
