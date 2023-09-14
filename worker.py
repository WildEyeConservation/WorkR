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
