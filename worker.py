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
from celery import Celery
from celery.signals import celeryd_after_setup

REDIS_IP = os.environ.get('REDIS_IP') or '127.0.0.1'
app = Celery('WorkR', broker='redis://'+REDIS_IP,backend='redis://'+REDIS_IP,broker_transport_options={'visibility_timeout': 86400},result_expires=86400,task_acks_late=True)
workername="default"

@app.task(name='WorkR.calculate_activity_pattern',bind=True,soft_time_limit=82800)
def calculate_activity_pattern(self,task_ids,trapgroups,groups,species,baseUnit,user_id,startDate,endDate,unit,centre,time,overlap, bucket, user_folder, csv, timeToIndependence, timeToIndependenceUnit):
    ''' Calculates the activity patterns for a set of species with R'''
    return {'status': None, 'error': None, 'activity_url': None}

@app.task(name='WorkR.calculate_occupancy_analysis',bind=True,soft_time_limit=82800)
def calculate_occupancy_analysis(self, task_ids,  species,  baseUnit,  trapgroups, groups, startDate, endDate,  window, siteCovs, detCovs, covOptions, user_id, user_folder, bucket, csv, timeToIndependence, timeToIndependenceUnit):
    ''' Calculates occupancy analysis'''
    return { 'status': None, 'error': None, 'occupancy_results': None }

@app.task(name='WorkR.calculate_spatial_capture_recapture',bind=True,soft_time_limit=82800)
def calculate_spatial_capture_recapture(self, species, user_id, task_ids, trapgroups, groups, startDate, endDate, window, tags, siteCovs, covOptions, bucket, user_folder, csv=False):
    ''' Calculates spatial capture recapture for a given species in R '''	
    return { 'status': None, 'error': None, 'scr_results': None }
