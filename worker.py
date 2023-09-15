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
from config import Config
from celery import Celery
from celery.signals import celeryd_after_setup
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pytz
import timezonefinder
import numpy as np
import utm
from app import db
from app.models import *
from sqlalchemy.sql import func, or_, alias, and_, distinct
import traceback
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import boto3

REDIS_IP = os.environ.get('REDIS_IP') or '127.0.0.1'
app = Celery('WorkR', broker='redis://'+REDIS_IP,backend='redis://'+REDIS_IP,broker_transport_options={'visibility_timeout': 86400},result_expires=86400,task_acks_late=True)
workername="default"

s3client = boto3.client('s3')
nothing_id = db.session.query(Label).filter(Label.description=='Nothing').first().id
knocked_id = db.session.query(Label).filter(Label.description=='Knocked Down').first().id
vhl_id = db.session.query(Label).filter(Label.description=='Vehicles/Humans/Livestock').first().id
unknown_id = db.session.query(Label).filter(Label.description=='Unknown').first().id
wrong_id = db.session.query(Label).filter(Label.description=='Wrong').first().id
remove_false_detections_id = db.session.query(Label).filter(Label.description=='Remove False Detections').first().id

def getChildList(label,task_id):
    '''Returns a list of all child label IDs for the specified label for the specified task.'''

    children = db.session.query(Label).filter(Label.parent_id==label.id).filter(Label.task_id==task_id).all()
    label_list = []
    for lab in children:
        label_list.append(lab.id)
        if db.session.query(Label).filter(Label.parent_id==lab.id).filter(Label.task_id==task_id).first():
            label_list.extend(getChildList(lab,task_id))
    return label_list

@app.task(name='WorkR.calculate_activity_pattern',bind=True,soft_time_limit=82800)
def calculate_activity_pattern(self,task_ids,trapgroups,groups,species,baseUnit,user_id,startDate,endDate,unit,centre,time,overlap, bucket, user_folder, csv, timeToIndependence, timeToIndependenceUnit):
    ''' Calculates the activity patterns for a set of species with R'''
    try:
        pandas2ri.activate()
        activity_url = None
        if task_ids:
            if task_ids[0] == '0':
                tasks = db.session.query(Task.id, Task.survey_id).join(Survey).filter(Survey.user_id == user_id).filter(Task.name != 'default').filter(~Task.name.contains('_o_l_d_')).filter(~Task.name.contains('_copying')).group_by(Task.survey_id).order_by(Task.id).all()
            else:
                tasks = db.session.query(Task.id, Task.survey_id).join(Survey).filter(Survey.user_id==user_id).filter(Task.id.in_(task_ids)).all()
            task_ids = [r[0] for r in tasks]
            survey_ids = list(set([r[1] for r in tasks]))

            if baseUnit == '1' or baseUnit == '4': # Image
                baseQuery = db.session.query(
                                Image.id,
                                Image.corrected_timestamp,
                                Label.id,
                                Label.description,
                                Trapgroup.tag,
                                Trapgroup.latitude,
                                Trapgroup.longitude
                            )\
                            .join(Detection)\
                            .join(Labelgroup)\
                            .join(Label,Labelgroup.labels)\
                            .join(Camera)\
                            .join(Trapgroup)\
                            .outerjoin(Sitegroup, Trapgroup.sitegroups)\
                            .filter(Labelgroup.task_id.in_(task_ids))\
                            .filter(or_(and_(Detection.source==model,Detection.score>Config.DETECTOR_THRESHOLDS[model]) for model in Config.DETECTOR_THRESHOLDS))\
                            .filter(Detection.static==False)\
                            .filter(~Detection.status.in_(['deleted','hidden']))\
                            .filter(Trapgroup.survey_id.in_(survey_ids))

            elif baseUnit == '2': # Cluster
                baseQuery = db.session.query(
                                Cluster.id,
                                Image.corrected_timestamp,
                                Label.id,
                                Label.description,
                                Trapgroup.tag,
                                Trapgroup.latitude,
                                Trapgroup.longitude
                            )\
                            .join(Image,Cluster.images)\
                            .join(Detection) \
                            .join(Labelgroup)\
                            .join(Label,Labelgroup.labels)\
                            .join(Camera)\
                            .join(Trapgroup)\
                            .outerjoin(Sitegroup, Trapgroup.sitegroups)\
                            .filter(Cluster.task_id.in_(task_ids))\
                            .filter(Labelgroup.task_id.in_(task_ids))\
                            .filter(or_(and_(Detection.source==model,Detection.score>Config.DETECTOR_THRESHOLDS[model]) for model in Config.DETECTOR_THRESHOLDS))\
                            .filter(Detection.static==False)\
                            .filter(~Detection.status.in_(['deleted','hidden']))\
                            .filter(Trapgroup.survey_id.in_(survey_ids))

            elif baseUnit == '3':  # Detection
                baseQuery = db.session.query(
                                Detection.id,
                                Image.corrected_timestamp,
                                Label.id,
                                Label.description,
                                Trapgroup.tag,
                                Trapgroup.latitude,
                                Trapgroup.longitude
                            )\
                            .join(Image)\
                            .join(Camera)\
                            .join(Trapgroup)\
                            .join(Labelgroup)\
                            .join(Label,Labelgroup.labels)\
                            .outerjoin(Sitegroup, Trapgroup.sitegroups)\
                            .filter(Labelgroup.task_id.in_(task_ids))\
                            .filter(or_(and_(Detection.source==model,Detection.score>Config.DETECTOR_THRESHOLDS[model]) for model in Config.DETECTOR_THRESHOLDS))\
                            .filter(Detection.static==False)\
                            .filter(~Detection.status.in_(['deleted','hidden']))\
                            .filter(Trapgroup.survey_id.in_(survey_ids))

            if trapgroups != '0' and trapgroups != '-1' and groups != '0' and groups != '-1':
                baseQuery = baseQuery.filter(or_(Trapgroup.id.in_(trapgroups), Sitegroup.id.in_(groups)))
            elif trapgroups != '0' and trapgroups != '-1':
                baseQuery = baseQuery.filter(Trapgroup.id.in_(trapgroups))
            elif groups != '0' and groups != '-1':
                baseQuery = baseQuery.filter(Sitegroup.id.in_(groups))

            # Filter by species and children
            parent_children = {}
            if species != '0':
                labels = db.session.query(Label).filter(Label.description.in_(species)).filter(Label.task_id.in_(task_ids)).all()
                label_list = []
                for label in labels:
                    label_list.append(label.id)
                    children = getChildList(label,int(label.task_id))
                    if children:
                        if label.description not in parent_children.keys():
                            parent_children[label.description] = []
                        parent_children[label.description].extend(children)
                    label_list.extend(children)
                baseQuery = baseQuery.filter(Labelgroup.labels.any(Label.id.in_(label_list)))
            else:
                vhl = db.session.query(Label).get(vhl_id)
                label_list = [vhl_id,nothing_id,knocked_id]
                for task_id in task_ids:
                    label_list.extend(getChildList(vhl,int(task_id)))
                baseQuery = baseQuery.filter(~Labelgroup.labels.any(Label.id.in_(label_list)))

            if startDate: baseQuery = baseQuery.filter(Image.corrected_timestamp >= startDate)

            if endDate: baseQuery = baseQuery.filter(Image.corrected_timestamp <= endDate)

            # TODO: Look at group_by usage & check query results in general
            # if baseUnit == '1' or baseUnit == '4':
            #     baseQuery = baseQuery.filter(Image.corrected_timestamp != None).order_by(Image.corrected_timestamp).group_by(Image.id).all()
            # elif baseUnit == '2':
            #     baseQuery = baseQuery.filter(Image.corrected_timestamp != None).order_by(Image.corrected_timestamp).group_by(Cluster.id).all()
            # elif baseUnit == '3':
            #     baseQuery = baseQuery.filter(Image.corrected_timestamp != None).order_by(Image.corrected_timestamp).group_by(Detection.id).all()

            baseQuery = baseQuery.filter(Image.corrected_timestamp != None).order_by(Image.corrected_timestamp).all()

            df = pd.DataFrame(baseQuery, columns=['id','timestamp','label_id','species', 'tag', 'latitude','longitude'])
            df.drop_duplicates()

            # Group rows by species and site and check if the time between rows is less than the timeToIndependence and remove rows if so
            if timeToIndependence:
                if timeToIndependenceUnit == 's':
                    timeToIndependence = int(timeToIndependence)
                elif timeToIndependenceUnit == 'm':
                    timeToIndependence = int(timeToIndependence) * 60
                elif timeToIndependenceUnit == 'h':
                    timeToIndependence = int(timeToIndependence) * 3600
                timeToIndependence = timedelta(seconds=timeToIndependence)
                
                df = df.sort_values(by=['species','tag', 'latitude', 'longitude','timestamp'])
                df['timedelta'] = df.groupby(['species','tag','latitude','longitude'])['timestamp'].diff()
                df['timedelta'] = df['timedelta'].fillna(timedelta(seconds=9999999))
                df = df[df['timedelta'] >= timeToIndependence]
                df = df.drop(columns=['timedelta'])

            # Get timezone from lat and lng coordinates
            lat = df['latitude'].unique().mean()
            lng = df['longitude'].unique().mean()

            tf = timezonefinder.TimezoneFinder()
            timezone = tf.timezone_at(lng=lng, lat=lat)
            if timezone:
                tz = timezone
                tz_obj = pytz.timezone(timezone)
                if df['timestamp'].iloc[0]:
                    tz_now = tz_obj.localize(df['timestamp'].iloc[0])
                else:
                    tz_now = tz_obj.localize(datetime.now())
                utc_offset_hours =  tz_now.utcoffset().total_seconds()/3600   
            else:
                utc_offset_hours = 0
                tz = 'UTC'

            # Change species description to parent species 
            if species == '0':
                df['species'] = 'All'
            else:
                for key, value in parent_children.items():
                    df.loc[df['label_id'].isin(value), 'species'] = key

            df = df[['id','timestamp','species']]

            if csv:
                # Convert to CSV and upload to bucket
                with tempfile.NamedTemporaryFile(delete=True, suffix='.csv') as temp_file:
                    df.to_csv(temp_file.name,index=False)
                    fileName = user_folder+'/docs/' + 'Activity_Pattern_CSV'
                    for specie in species:
                        fileName += '_' + specie
                    fileName += '.csv'
                    s3client.put_object(Bucket=bucket,Key=fileName,Body=temp_file)
                    activity_url = "https://"+ bucket + ".s3.amazonaws.com/" + fileName

                    # Schedule deletion
                    # deleteFile.apply_async(kwargs={'fileName': fileName}, countdown=21600)

            else: 
                if len(df)> 0:
                    # Convert to R dataframe and run R script and upload to bucket
                    r_df = robjects.conversion.py2rpy(df)
                    r = robjects.r
                    r.source('R/activity_pattern.R')

                    with tempfile.NamedTemporaryFile(delete=True, suffix='.JPG') as temp_file:
                        fileName = user_folder+'/docs/' + 'Activity_Pattern' 
                        for specie in species:
                            fileName += '_' + specie
                        fileName += '_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.JPG'
                        file_name = temp_file.name.split('.JPG')[0]
                        if species == '0':
                            species_r = robjects.StrVector(['All'])
                        else:
                            species_r = robjects.StrVector(species)
                        lat = robjects.FloatVector([lat])
                        lng = robjects.FloatVector([lng])
                        r.calculate_activity_pattern(r_df,file_name,species_r,centre,unit,time,overlap,lat,lng,utc_offset_hours,tz)
                        temp_file = open(temp_file.name, 'rb')
                        s3client.put_object(Bucket=bucket,Key=fileName,Body=temp_file)
                        activity_url = "https://"+ bucket + ".s3.amazonaws.com/" + fileName

                        # Schedule deletion
                        # deleteFile.apply_async(kwargs={'fileName': fileName}, countdown=21600)
                else:
                    activity_url = None

        status = 'SUCCESS'
        error = None

    except Exception as exc:
        print(' ')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(traceback.format_exc())
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(' ')
        status = 'FAILURE' 
        error = str(exc)

    finally:
        db.session.remove()

    return {'status': status, 'error': error, 'activity_url': activity_url}

@app.task(name='WorkR.calculate_occupancy_analysis',bind=True,soft_time_limit=82800)
def calculate_occupancy_analysis(self, task_ids,  species,  baseUnit,  trapgroups, groups, startDate, endDate,  window, siteCovs, detCovs, covOptions, user_id, user_folder, bucket, csv, timeToIndependence, timeToIndependenceUnit):
    ''' Calculates occupancy analysis'''
    try:
        pandas2ri.activate()
        occupancy_results = {}

        if task_ids:
            if task_ids[0] == '0':
                tasks = db.session.query(Task.id, Task.survey_id).join(Survey).filter(Survey.user_id == user_id).filter(Task.name != 'default').filter(~Task.name.contains('_o_l_d_')).filter(~Task.name.contains('_copying')).group_by(Task.survey_id).order_by(Task.id).all()
            else:
                tasks = db.session.query(Task.id, Task.survey_id).join(Survey).filter(Survey.user_id == user_id).filter(Task.id.in_(task_ids)).all()

            task_ids = [r[0] for r in tasks]
            survey_ids = list(set([r[1] for r in tasks]))

            # Get site data for site operation
            siteQuery =  db.session.query(
                Trapgroup.id,
                Trapgroup.tag,
                Trapgroup.latitude,
                Trapgroup.longitude,
                Trapgroup.altitude, 
                func.min(Image.corrected_timestamp),
                func.max(Image.corrected_timestamp),
            )\
            .join(Camera, Trapgroup.id == Camera.trapgroup_id)\
            .join(Image)\
            .outerjoin(Sitegroup, Trapgroup.sitegroups)\
            .filter(Trapgroup.survey_id.in_(survey_ids))\
            .filter(Image.corrected_timestamp != None)\
            .group_by(Trapgroup.id)

            if trapgroups != '0' and trapgroups != '-1' and groups != '0' and groups != '-1':
                siteQuery = siteQuery.filter(or_(Trapgroup.id.in_(trapgroups), Sitegroup.id.in_(groups)))
            elif trapgroups != '0' and trapgroups != '-1':
                siteQuery = siteQuery.filter(Trapgroup.id.in_(trapgroups))
            elif groups != '0' and groups != '-1':
                siteQuery = siteQuery.filter(Sitegroup.id.in_(groups))

            site_df = pd.DataFrame(siteQuery.all(), columns=['id', 'site_tag', 'latitude', 'longitude', 'altitude', 'first_date', 'last_date'])

            setup_col = 'first_date'
            retrieval_col = 'last_date'
            station_col = 'site_id'

            # Combine cameras with same id and assign the min first_date and max last_date for each site
            site_df = site_df.groupby(['site_tag', 'latitude', 'longitude']).agg({'first_date': 'min', 'last_date': 'max'}).reset_index()

            # Add a id column that is combined from the site_tag and coordinates
            site_df['site_id'] = (
                site_df['site_tag'] + '_' +
                site_df['latitude'].apply(lambda lat: f'{lat:.4f}') + '_' +
                site_df['longitude'].apply(lambda lng: f'{lng:.4f}')
            )

            # Convert first_date and last_date ymd format 
            site_df['first_date'] = pd.to_datetime(site_df['first_date']).dt.strftime('%Y-%m-%d')
            site_df['last_date'] = pd.to_datetime(site_df['last_date']).dt.strftime('%Y-%m-%d')

            if baseUnit == '1' or baseUnit == '4': # Image
                baseQuery = db.session.query(
                    Image.id,
                    Image.corrected_timestamp,
                    Trapgroup.tag,
                    Trapgroup.latitude,
                    Trapgroup.longitude,
                    Label.description,
                )\
                .join(Detection)\
                .join(Labelgroup)\
                .join(Label, Labelgroup.labels)\
                .join(Camera)\
                .join(Trapgroup)\
                .outerjoin(Sitegroup, Trapgroup.sitegroups)\
                .filter(Trapgroup.survey_id.in_(survey_ids))\
                .filter(Image.corrected_timestamp != None)\
                .filter(~Detection.status.in_(['deleted','hidden']))\
                .filter(or_(and_(Detection.source==model,Detection.score>Config.DETECTOR_THRESHOLDS[model]) for model in Config.DETECTOR_THRESHOLDS))\
                .filter(Detection.static==False)\
                .filter(Labelgroup.task_id.in_(task_ids))

            elif baseUnit == '2': # Cluster
                baseQuery = db.session.query(
                    Cluster.id,
                    Image.corrected_timestamp,
                    Trapgroup.tag,
                    Trapgroup.latitude,
                    Trapgroup.longitude,
                    Label.description,
                )\
                .join(Image,Cluster.images)\
                .join(Camera)\
                .join(Trapgroup)\
                .join(Detection)\
                .join(Labelgroup)\
                .join(Label, Labelgroup.labels)\
                .outerjoin(Sitegroup, Trapgroup.sitegroups)\
                .filter(Trapgroup.survey_id.in_(survey_ids))\
                .filter(Image.corrected_timestamp != None)\
                .filter(~Detection.status.in_(['deleted','hidden']))\
                .filter(or_(and_(Detection.source==model,Detection.score>Config.DETECTOR_THRESHOLDS[model]) for model in Config.DETECTOR_THRESHOLDS))\
                .filter(Detection.static==False)\
                .filter(Labelgroup.task_id.in_(task_ids))\
                .filter(Cluster.task_id.in_(task_ids))

            elif baseUnit == '3':  # Detection
                baseQuery = db.session.query(
                    Detection.id,
                    Image.corrected_timestamp,
                    Trapgroup.tag,
                    Trapgroup.latitude,
                    Trapgroup.longitude,
                    Label.description,
                )\
                .join(Image)\
                .join(Camera)\
                .join(Trapgroup)\
                .join(Labelgroup)\
                .join(Label, Labelgroup.labels)\
                .outerjoin(Sitegroup, Trapgroup.sitegroups)\
                .filter(Trapgroup.survey_id.in_(survey_ids))\
                .filter(Image.corrected_timestamp != None)\
                .filter(~Detection.status.in_(['deleted','hidden']))\
                .filter(or_(and_(Detection.source==model,Detection.score>Config.DETECTOR_THRESHOLDS[model]) for model in Config.DETECTOR_THRESHOLDS))\
                .filter(Detection.static==False)\
                .filter(Labelgroup.task_id.in_(task_ids))
                
            if startDate: baseQuery = baseQuery.filter(Image.corrected_timestamp >= startDate)

            if endDate: baseQuery = baseQuery.filter(Image.corrected_timestamp <= endDate)

            if species != '0':
                labels = db.session.query(Label).filter(Label.description.in_(species)).filter(Label.task_id.in_(task_ids)).all()
                label_list = []
                for label in labels:
                    label_list.append(label.id)
                    label_list.extend(getChildList(label,int(label.task_id)))
                baseQuery = baseQuery.filter(Labelgroup.labels.any(Label.id.in_(label_list)))
            else:
                vhl = db.session.query(Label).get(vhl_id)
                label_list = [vhl_id,nothing_id,knocked_id]
                for task_id in task_ids:
                    label_list.extend(getChildList(vhl,int(task_id)))
                baseQuery = baseQuery.filter(~Labelgroup.labels.any(Label.id.in_(label_list)))

            if trapgroups != '0' and trapgroups != '-1' and groups != '0' and groups != '-1':
                baseQuery = baseQuery.filter(or_(Trapgroup.id.in_(trapgroups), Sitegroup.id.in_(groups)))
            elif trapgroups != '0' and trapgroups != '-1':
                baseQuery = baseQuery.filter(Trapgroup.id.in_(trapgroups))
            elif groups != '0' and groups != '-1':
                baseQuery = baseQuery.filter(Sitegroup.id.in_(groups))

            # TODO: double check group_by
            if baseUnit == '1' or baseUnit == '4': # Image
                baseQuery = baseQuery.group_by(Image.id).all()
            elif baseUnit == '2': # Cluster
                baseQuery = baseQuery.group_by(Cluster.id).all()
            elif baseUnit == '3':  # Detection
                baseQuery = baseQuery.group_by(Detection.id).all()
            
            detection_df = pd.DataFrame(baseQuery, columns=['id','timestamp', 'site_tag', 'latitude', 'longitude', 'species'])

            # Add site_id to be a combination of site_tag and coordinates
            detection_df['site_id'] = (
            detection_df['site_tag'] + '_' +
            detection_df['latitude'].apply(lambda lat: f'{lat:.4f}') + '_' +
            detection_df['longitude'].apply(lambda lng: f'{lng:.4f}'))

            # Group rows by species and site and check if the time between rows is less than the timeToIndependence and remove rows if so
            if timeToIndependence:
                if timeToIndependenceUnit == 's':
                    timeToIndependence = int(timeToIndependence)
                elif timeToIndependenceUnit == 'm':
                    timeToIndependence = int(timeToIndependence) * 60
                elif timeToIndependenceUnit == 'h':
                    timeToIndependence = int(timeToIndependence) * 3600
                timeToIndependence = timedelta(seconds=timeToIndependence)
                
                detection_df = detection_df.sort_values(by=['species','site_id','timestamp'])
                detection_df['timedelta'] = detection_df.groupby(['species','site_id'])['timestamp'].diff()
                detection_df['timedelta'] = detection_df['timedelta'].fillna(timedelta(seconds=9999999))
                detection_df = detection_df[detection_df['timedelta'] >= timeToIndependence]
                detection_df = detection_df.drop(columns=['timedelta'])

            # add a column called Date that is the date of the timestamp in ymd
            detection_df['date'] = pd.to_datetime(detection_df['timestamp']).dt.strftime('%Y-%m-%d')

            if species != '0' and len(species) == 1:
                detection_df['species'] = species[0]
                species = species[0]
            else:
                detection_df['species'] = 'Animal'
                species = 'Animal'

            if len(siteCovs) > 0:
                # Create df of siteCovs
                site_cov = pd.DataFrame(siteCovs)
                site_cov = site_cov.rename(columns={'covariate': 'site_id'}).set_index('site_id').transpose()

                # Remove rows which site_id is not in the site_df
                site_cov = site_cov[site_cov.index.isin(site_df['site_id'])]

                # rename site_id to index
                site_cov = site_cov.rename_axis('site_id').reset_index()
                site_cov = site_cov.rename_axis(None, axis=1)
            else:
                site_cov = pd.DataFrame()

            if len(detCovs) > 0:
                # Create df of detCovs
                det_cov = pd.DataFrame(detCovs)
                det_cov = det_cov.rename(columns={'covariate': 'site_id'}).set_index('site_id').transpose()

                # Remove rows which site_id is not in the site_df
                det_cov = det_cov[det_cov.index.isin(site_df['site_id'])]

                # rename site_id to index
                det_cov = det_cov.rename_axis('site_id').reset_index()
                det_cov = det_cov.rename_axis(None, axis=1)
            else:
                det_cov = pd.DataFrame()

            if len(covOptions) > 0:
                cov_options = pd.DataFrame(covOptions)
                # Use covariate column as index
                cov_options = cov_options.set_index('covariate')
            else:
                cov_options = pd.DataFrame()

            # all cov df
            if len(site_cov) > 0 and len(det_cov) > 0:
                all_cov = pd.merge(site_cov, det_cov, on='site_id')
            elif len(site_cov) > 0:
                all_cov = site_cov.copy()
            elif len(det_cov) > 0:
                all_cov = det_cov.copy()
            else:
                all_cov = pd.DataFrame()

            if csv:
                dfs = [detection_df, site_cov, det_cov, all_cov, site_df, cov_options]
                dfs_names = ['detection', 'site_cov', 'det_cov', 'all_cov', 'site', 'cov_options']
                occu_urls = []
                for i in range(len(dfs)):
                    with tempfile.NamedTemporaryFile(delete=True, suffix='.csv') as temp_file:
                        df = dfs[i]
                        df_name = dfs_names[i]
                        if df_name == 'cov_options':
                            df.to_csv(temp_file.name, index=True)
                        else:
                            df.to_csv(temp_file.name, index=False)
                        fileName = user_folder+'/docs/' + 'Occupancy' + '_' + species + '_' + df_name + '.csv'
                        s3client.put_object(Bucket=bucket,Key=fileName,Body=temp_file)
                        occupancy_url = "https://"+ bucket + ".s3.amazonaws.com/" + fileName
                        print(occupancy_url)
                        occu_urls.append(occupancy_url)

                        # Schedule deletion
                        # deleteFile.apply_async(kwargs={'fileName': fileName}, countdown=21600)

                occupancy_results = {
                    'csv_urls': occu_urls
                }

            else: 
                if len(detection_df) > 0:
                    # Convert to R dataframe       
                    detection_df_r = robjects.conversion.py2rpy(detection_df)
                    site_cov_r = robjects.conversion.py2rpy(site_cov)
                    det_cov_r = robjects.conversion.py2rpy(det_cov)
                    all_cov_r = robjects.conversion.py2rpy(all_cov)
                    site_df_r = robjects.conversion.py2rpy(site_df)
                    cov_options_r = robjects.conversion.py2rpy(cov_options)

                    # call the occupancy function
                    r = robjects.r
                    r.source('R/occupancy.R')
                    occupancy_results = r.occupancy(detection_df_r, site_df_r, setup_col, retrieval_col, station_col, window, site_cov_r, det_cov_r, all_cov_r, species, cov_options_r)

                    nr_plots = int(occupancy_results.rx2('nr_plots')[0])
                    best_model_cov_names = list(occupancy_results.rx2('best_model_cov_names'))
                    naive_occupancy = float(occupancy_results.rx2('naive_occu')[0])
                    total_sites = int(occupancy_results.rx2('total_sites')[0])
                    total_sites_occupied = int(occupancy_results.rx2('total_sites_occupied')[0])
                    model_name = str(occupancy_results.rx2('model_sel_name')[0])
                    model_formula = str(occupancy_results.rx2('best_model_formula')[0])
                    occu_est = float(occupancy_results.rx2('occu_est')[0])
                    det_est = float(occupancy_results.rx2('det_est')[0])

                    aic = pd.DataFrame(occupancy_results.rx2('aic'))
                    aic = aic.replace([np.inf, -np.inf], 'Inf')
                    aic = aic.replace([np.nan], 'NA')
                    aic = aic.to_dict(orient='records')

                    best_model_summary_state = pd.DataFrame(occupancy_results.rx2('best_model_summary_state')) 
                    best_model_summary_state = best_model_summary_state.replace([np.inf, -np.inf], 'Inf')
                    best_model_summary_state = best_model_summary_state.replace([np.nan], 'NA')
                    best_model_summary_state = best_model_summary_state.to_dict(orient='records')

                    best_model_summary_det = pd.DataFrame(occupancy_results.rx2('best_model_summary_det'))
                    best_model_summary_det = best_model_summary_det.replace([np.inf, -np.inf], 'Inf')
                    best_model_summary_det = best_model_summary_det.replace([np.nan], 'NA')
                    best_model_summary_det = best_model_summary_det.to_dict(orient='records')

                    predict_tables = []
                    # Get the temp files
                    if len(best_model_cov_names) > 0:
                        occu_files = []
                        for i in range(len(best_model_cov_names)):
                            occu = {}
                            occu['name'] = best_model_cov_names[i]
                            occu['images'] = []
                            for j in range(nr_plots):
                                with tempfile.NamedTemporaryFile(delete=True, suffix='.JPG') as temp_file:
                                    fileName = user_folder+'/docs/' + 'Occupancy' + '_' + species + '_' + best_model_cov_names[i] + '_' + str(j+1)
                                    fileName += '_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.JPG'
                                    file_name = temp_file.name.split('.JPG')[0]
                                    pred = r.plot_occupancy(j+1, file_name, best_model_cov_names[i])
                                    temp_file = open(temp_file.name, 'rb')
                                    s3client.put_object(Bucket=bucket,Key=fileName,Body=temp_file)
                                    occupancy_url = "https://"+ bucket + ".s3.amazonaws.com/" + fileName
                                    occu['images'].append(occupancy_url)

                                    pred_table = pd.DataFrame(pred.rx2('prediction_table')) 
                                    pred_type = str(pred.rx2('type')[0])
                                    if len(pred_table) > 0:
                                        pred_table = pred_table.replace([np.inf, -np.inf], 'Inf')
                                        pred_table = pred_table.replace([np.nan], 'NA')
                                        pred_table = pred_table.to_dict(orient='records')
                                        predict_tables.append({
                                            'name': best_model_cov_names[i],
                                            'type': pred_type,
                                            'table': pred_table
                                        })

                                    # Schedule deletion
                                    # deleteFile.apply_async(kwargs={'fileName': fileName}, countdown=21600)
                            occu_files.append(occu)

                    else:
                        occu_files = []
                        if model_name == "~1 ~ 1":
                            occu = {}
                            occu['name'] = model_name
                            occu['images'] = []
                            for j in range(nr_plots):
                                with tempfile.NamedTemporaryFile(delete=True, suffix='.JPG') as temp_file:
                                    fileName = user_folder+'/docs/' + 'Occupancy' + '_' + species + '_' + model_name + '_' + str(j+1)
                                    fileName += '_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.JPG'
                                    file_name = temp_file.name.split('.JPG')[0]
                                    pred = r.plot_occupancy(j+1, file_name, model_name)
                                    temp_file = open(temp_file.name, 'rb')
                                    s3client.put_object(Bucket=bucket,Key=fileName,Body=temp_file)
                                    occupancy_url = "https://"+ bucket + ".s3.amazonaws.com/" + fileName
                                    occu['images'].append(occupancy_url)

                                    pred_table = pd.DataFrame(pred.rx2('prediction_table')) 
                                    pred_type = str(pred.rx2('type')[0])
                                    if len(pred_table) > 0:
                                        pred_table = pred_table.replace([np.inf, -np.inf], 'Inf')
                                        pred_table = pred_table.replace([np.nan], 'NA')
                                        pred_table = pred_table.to_dict(orient='records')
                                        predict_tables.append({
                                            'name': model_name,
                                            'type': pred_type,
                                            'table': pred_table
                                        })

                                    # Schedule deletion
                                    # deleteFile.apply_async(kwargs={'fileName': fileName}, countdown=21600)
                            occu_files.append(occu)

                    occupancy_results = {
                        'naive_occupancy': naive_occupancy,
                        'total_sites': total_sites,
                        'total_sites_occupied': total_sites_occupied,
                        'occu_files': occu_files,
                        'model_formula': model_formula,
                        'aic': aic,
                        'best_model_summary_state': best_model_summary_state,
                        'best_model_summary_det': best_model_summary_det,
                        'occu_est': occu_est,
                        'det_est': det_est,
                        'predict_tables': predict_tables
                    }

                else:
                    occupancy_results = {
                        'naive_occupancy': 0,
                        'total_sites': 0,
                        'total_sites_occupied': 0,
                        'occu_files': [],
                        'model_formula': '',
                        'aic': [],
                        'best_model_summary_state': [],
                        'best_model_summary_det': [],
                        'occu_est': 0,
                        'det_est': 0,
                        'predict_tables': []
                    }

        status = 'SUCCESS'
        error = None

    except Exception as exc:
        print(' ')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(traceback.format_exc())
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(' ')
        status = 'FAILURE'
        error = str(exc)

    finally:
        db.session.remove()

    return { 'status': status, 'error': error, 'occupancy_results': occupancy_results }

@app.task(name='WorkR.calculate_spatial_capture_recapture',bind=True,soft_time_limit=82800)
def calculate_spatial_capture_recapture(self, species, user_id, task_ids, trapgroups, groups, startDate, endDate, window, tags, siteCovs, covOptions, bucket, user_folder, csv=False):
    ''' Calculates spatial capture recapture for a given species in R '''	
    try:
        pandas2ri.activate()
        results = {}
        status = None
        error = None
        temp_files = []

        if task_ids:
            if task_ids[0] == '0':
                tasks = db.session.query(Task.id, Task.survey_id).join(Survey).filter(Survey.user_id == user_id).filter(Task.name != 'default').filter(~Task.name.contains('_o_l_d_')).filter(~Task.name.contains('_copying')).group_by(Task.survey_id).order_by(Task.id).all()
            else:
                tasks = db.session.query(Task.id, Task.survey_id).join(Survey).filter(Survey.user_id == user_id).filter(Task.id.in_(task_ids)).all()

            task_ids = [r[0] for r in tasks]
            survey_ids = list(set([r[1] for r in tasks]))

            sites = db.session.query(Trapgroup.id, 
                                Trapgroup.tag, 
                                Trapgroup.latitude, 
                                Trapgroup.longitude,
                                func.min(Image.corrected_timestamp),
                                func.max(Image.corrected_timestamp))\
                            .join(Camera, Camera.trapgroup_id==Trapgroup.id)\
                            .join(Image)\
                            .outerjoin(Sitegroup, Trapgroup.sitegroups)\
                            .filter(Trapgroup.survey_id.in_(survey_ids))

    
            individuals = db.session.query(Individual.id,
                                                Individual.species,
                                                Tag.description,
                                                Image.corrected_timestamp,
                                                Trapgroup.tag,
                                                Trapgroup.latitude,
                                                Trapgroup.longitude)\
                                        .join(Detection,Individual.detections)\
                                        .join(Image)\
                                        .join(Task,Individual.tasks)\
                                        .outerjoin(Tag, Individual.tags)\
                                        .join(Camera)\
                                        .join(Trapgroup)\
                                        .outerjoin(Sitegroup, Trapgroup.sitegroups)\
                                        .filter(Task.id.in_(task_ids))\
                                        .filter(Individual.name!='unidentifiable')\
                                        .filter(Individual.active==True)\
                                        .filter(Trapgroup.survey_id.in_(survey_ids))\
                                        .filter(Image.corrected_timestamp!=None)

            if species !='0': individuals = individuals.filter(Individual.species.in_(species))

            if trapgroups != '0' and trapgroups != '-1' and groups != '0' and groups != '-1':
                individuals = individuals.filter(or_(Trapgroup.id.in_(trapgroups), Sitegroup.id.in_(groups)))
                sites = sites.filter(or_(Trapgroup.id.in_(trapgroups), Sitegroup.id.in_(groups)))
            elif trapgroups != '0' and trapgroups != '-1':
                individuals = individuals.filter(Trapgroup.id.in_(trapgroups))
                sites = sites.filter(Trapgroup.id.in_(trapgroups))
            elif groups != '0' and groups != '-1':
                individuals = individuals.filter(Sitegroup.id.in_(groups))
                sites = sites.filter(Sitegroup.id.in_(groups))

            if startDate: 
                individuals = individuals.filter(Image.corrected_timestamp >= startDate)
                sites = sites.filter(Image.corrected_timestamp >= startDate)

            if endDate: 
                individuals = individuals.filter(Image.corrected_timestamp <= endDate)       
                sites = sites.filter(Image.corrected_timestamp <= endDate)          


            individuals_df = pd.DataFrame(individuals.all(), columns=['individual_id', 'species', 'indiv_tags' ,'timestamp', 'tag', 'latitude', 'longitude'])   
            sites_df = pd.DataFrame(sites.group_by(Trapgroup.id).all(), columns=['id', 'tag', 'latitude', 'longitude', 'first_date', 'last_date'])

            if tags != '-1':
                individuals_df['indiv_tags'] = individuals_df['indiv_tags'].fillna('NA')
                tags.append('NA') 
                individuals_df = individuals_df[individuals_df['indiv_tags'].isin(tags)]

            # add column for session and set to 1
            individuals_df['session'] = 1

            # Add site_id column
            individuals_df['site_id'] = (individuals_df['tag'] + '_' + individuals_df['latitude'].apply(lambda lat: f'{lat:.4f}') + '_' + individuals_df['longitude'].apply(lambda lng: f'{lng:.4f}'))

            sites_df = sites_df.groupby(['tag', 'latitude', 'longitude']).agg({'first_date': 'min', 'last_date': 'max'}).reset_index()
            sites_df['site_id'] = (sites_df['tag'] + '_' + sites_df['latitude'].apply(lambda lat: f'{lat:.4f}') + '_' + sites_df['longitude'].apply(lambda lng: f'{lng:.4f}'))

            # create individaul count df that has all sites same as sites_df and counts set to 0
            individual_counts = sites_df[['site_id']].copy()
            individual_counts['count'] = 0

            # Add UTM X and Y columns to sites_df and scale to km
            sites_df['utm_x'] = sites_df.apply(lambda row: utm.from_latlon(row['latitude'], row['longitude'])[0]/1000, axis=1)
            sites_df['utm_y'] = sites_df.apply(lambda row: utm.from_latlon(row['latitude'], row['longitude'])[1]/1000, axis=1)

            # get grid zone and zone number
            utm_mean = utm.from_latlon(sites_df['latitude'].mean(), sites_df['longitude'].mean())
            zone_number = utm_mean[2]
            grid_zone = utm_mean[3]

            # Add a date column
            individuals_df['date'] = individuals_df['timestamp'].dt.date
            sites_df['first_date'] = sites_df['first_date'].dt.date
            sites_df['last_date'] = sites_df['last_date'].dt.date

            # Get min_date and max_date
            if startDate:
                min_date = datetime.strptime(startDate.split(' ')[0], '%Y-%m-%d').date()
            else:
                min_date = individuals_df['timestamp'].min()

            if endDate:
                max_date = datetime.strptime(endDate.split(' ')[0], '%Y-%m-%d').date()
            else:
                max_date = individuals_df['timestamp'].max()

            # set df_dh to unique individuals
            df_dh = individuals_df[['individual_id']].copy()
            df_dh = df_dh.drop_duplicates(subset=['individual_id'])

            # Create a list of dates (YYYY-MM-DD) based on the window (no time)
            dates = pd.date_range(min_date, max_date, freq=f'{window}D').date

            for i, date in enumerate(dates):
                col_name = 'occasion.' + str(i+1)
                sites_df[col_name] = 0
                df_dh[col_name] = 0
                site_date_filter = (sites_df['first_date'] <= date) & (sites_df['last_date'] >= date)
                sites_df.loc[site_date_filter, col_name] = 1

                if i < len(dates)-1:
                    indiviudal_date_filter = (individuals_df['date'] >= date) & (individuals_df['date'] < dates[i+1])
                else:
                    indiviudal_date_filter = (individuals_df['date'] >= date)
                individuals_df.loc[indiviudal_date_filter, 'occasion'] = i+1

                df_dh.loc[df_dh['individual_id'].isin(individuals_df[indiviudal_date_filter]['individual_id']), col_name] = 1
                
            individuals_df['occasion'] = individuals_df['occasion'].astype(int)


            # Only keep entries where there is unique individual_id, occasion, and site_id
            individuals_df = individuals_df.drop_duplicates(subset=['individual_id', 'occasion', 'site_id'])

            # Only keep columns: session, occasion, site_id, individual_id
            if tags != '-1':
                edf = individuals_df[['session', 'occasion', 'site_id', 'individual_id', 'indiv_tags']].copy()

                # If all tags are NA then remove the indiv_tags column
                if len(individuals_df['indiv_tags'].unique()) == 1 and individuals_df['indiv_tags'].unique()[0] == 'NA':
                    edf = edf.drop(columns=['indiv_tags'])
                    tags = '-1'
            else:
                edf = individuals_df[['session', 'occasion', 'site_id', 'individual_id']].copy()

            # Only keep columns: site_id, utm_x, utm_y,  occasion.1, occasion.2, occasion.3, etc.
            tdf = sites_df[['site_id', 'utm_x', 'utm_y'] + [col for col in sites_df.columns if 'occasion' in col]].copy()

            # drop sites that only has 0s in occasion columns (sites that weren't open during the survey)
            tdf = tdf[tdf[[col for col in tdf.columns if 'occasion' in col]].sum(axis=1) > 0]

            if len(siteCovs) > 0:
                # Create df of siteCovs
                site_cov = pd.DataFrame(siteCovs)
                site_cov = site_cov.rename(columns={'covariate': 'site_id'}).set_index('site_id').transpose()

                # Remove rows which site_id is not in the site_df
                site_cov = site_cov[site_cov.index.isin(tdf['site_id'])]

                # rename site_id to index
                site_cov = site_cov.rename_axis('site_id').reset_index()
                site_cov = site_cov.rename_axis(None, axis=1)

                tdf['sep'] = '/'
                
                #merge site_cov with tdf
                tdf = pd.merge(tdf, site_cov, on='site_id', how='left')

                # get all col names except site_id
                cov_names = [col for col in site_cov.columns if col != 'site_id']

            else:
                site_cov = pd.DataFrame()
                cov_names = []

            if len(covOptions) > 0:
                cov_options = pd.DataFrame(covOptions)
                cov_options = cov_options.set_index('covariate')
            else:
                cov_options = pd.DataFrame()


            # count number of unique individuals per site 
            site_grouped = edf.groupby('site_id')['individual_id'].nunique()
            for site_id, count in site_grouped.items():
                mask = individual_counts['site_id'] == site_id
                individual_counts.loc[mask, 'count'] = count

            max_count = individual_counts['count'].max()
            individual_counts = individual_counts.to_dict(orient='records')
            individual_counts.append({'max_count': str(max_count)})


            if csv:
                dfs = [edf, tdf, df_dh]
                dfs_names = ['edf', 'tdf', 'dh']
                scr_urls = []
                for i in range(len(dfs)):
                    with tempfile.NamedTemporaryFile(delete=True, suffix='.csv') as temp_file:
                        df = dfs[i]
                        df_name = dfs_names[i]
                        df.to_csv(temp_file.name, index=False)
                        fileName = user_folder+'/docs/' + 'SCR' + '_' 
                        for specie in species:
                            fileName += specie + '_'
                        fileName += df_name + '.csv'
                        s3client.put_object(Bucket=bucket,Key=fileName,Body=temp_file)
                        scr_url = "https://"+ bucket + ".s3.amazonaws.com/" + fileName
                        print(scr_url)
                        scr_urls.append(scr_url)

                        # Schedule deletion
                        # deleteFile.apply_async(kwargs={'fileName': fileName}, countdown=21600)

                results = scr_urls
                status = 'SUCCESS'
                error = None

            else:
                # Convert to R dataframes
                edf_r = robjects.conversion.py2rpy(edf)
                tdf_r = robjects.conversion.py2rpy(tdf)
                cov_names_r = robjects.conversion.py2rpy(cov_names)
                cov_options_r = robjects.conversion.py2rpy(cov_options)
                df_dh_r = robjects.conversion.py2rpy(df_dh)

                session_col = 'session'
                id_col = 'individual_id'
                occ_col = 'occasion'
                site_col = 'site_id'
                if tags != '-1':
                    tag_col = 'indiv_tags'	
                else:
                    tag_col = 'none'
                sep = '/'

                # Run the R function	
                r = robjects.r
                r.source('R/spatial_capture_recapture.R')

                plots = ['captures', 'state-space', 'density_map']
                temp_files = []
                temp_file_names = []
                for i in range(len(plots)):
                    temp_files.append(tempfile.NamedTemporaryFile(suffix='.JPG', delete=True))
                    temp_file_names.append(temp_files[i].name.split('.JPG')[0])

                # file_names_r = robjects.StrVector(temp_file_names)
                file_names_r = robjects.conversion.py2rpy(temp_file_names)

                # Run the R function
                scr_results = r.spatial_capture_recapture(edf_r, tdf_r, session_col, id_col, occ_col, site_col, tag_col, sep, cov_names_r, cov_options_r, df_dh_r, file_names_r)

                density = pd.DataFrame(scr_results.rx2('density'))
                if density.empty:
                    density = []
                else:
                    if 'sex' in density.columns:
                        density.rename(columns={'estimate': 'Estimate', 'se': 'Standard Error', 'lwr': 'Lower Bound', 'upr': 'Upper Bound', 'sex': 'Sex'}, inplace=True)
                        density = density[['Estimate', 'Standard Error', 'Lower Bound', 'Upper Bound', 'Sex']]
                    else:
                        density.rename(columns={'estimate': 'Estimate', 'se': 'Standard Error', 'lwr': 'Lower Bound', 'upr': 'Upper Bound'}, inplace=True)
                        density = density[['Estimate', 'Standard Error', 'Lower Bound', 'Upper Bound']]
                    density = density.replace([np.inf, -np.inf], 'Inf')
                    density = density.replace([np.nan], 'NA')
                    density = density.to_dict(orient='records')

                abundance = pd.DataFrame(scr_results.rx2('abundance'))
                if abundance.empty:
                    abundance = []
                else:
                    if 'sex' in abundance.columns:
                        abundance.rename(columns={'estimate': 'Estimate', 'se': 'Standard Error', 'lwr': 'Lower Bound', 'upr': 'Upper Bound', 'sex': 'Sex', 'state_space': 'State space (km2)'}, inplace=True)
                        abundance = abundance[['Estimate', 'Standard Error', 'Lower Bound', 'Upper Bound', 'Sex','State space (km2)']]
                    else:
                        abundance.rename(columns={'estimate': 'Estimate', 'se': 'Standard Error', 'lwr': 'Lower Bound', 'upr': 'Upper Bound', 'state_space': 'State space (km2)'}, inplace=True)
                        abundance = abundance[['Estimate', 'Standard Error', 'Lower Bound', 'Upper Bound', 'State space (km2)']]

                    abundance = abundance.replace([np.inf, -np.inf], 'Inf')
                    abundance = abundance.replace([np.nan], 'NA')
                    abundance = abundance.to_dict(orient='records')

                det_prob = pd.DataFrame(scr_results.rx2('det_prob'))
                if det_prob.empty:
                    det_prob = []
                else:
                    if 'sex' in det_prob.columns:
                        det_prob.rename(columns={'estimate': 'Estimate', 'se': 'Standard Error', 'lwr': 'Lower Bound', 'upr': 'Upper Bound', 'sex': 'Sex'}, inplace=True)
                        det_prob = det_prob[['Estimate', 'Standard Error', 'Lower Bound', 'Upper Bound', 'Sex']]
                    else:
                        det_prob.rename(columns={'estimate': 'Estimate', 'se': 'Standard Error', 'lwr': 'Lower Bound', 'upr': 'Upper Bound'}, inplace=True)
                        det_cols = ['estimate', 'standard error', 'lower bound', 'upper bound', 'session']
                        det_prob = det_prob[['Estimate', 'Standard Error', 'Lower Bound', 'Upper Bound'] + [col for col in det_prob.columns if col.lower() not in det_cols]]
                    det_prob = det_prob.replace([np.inf, -np.inf], 'Inf')
                    det_prob = det_prob.replace([np.nan], 'NA')
                    det_prob = det_prob.to_dict(orient='records')

                sigma = pd.DataFrame(scr_results.rx2('sigma'))
                if sigma.empty:
                    sigma = []
                else:          
                    if 'sex' in sigma.columns:
                        sigma.rename(columns={'estimate': 'Estimate', 'se': 'Standard Error', 'lwr': 'Lower Bound', 'upr': 'Upper Bound', 'sex': 'Sex'}, inplace=True)
                        sigma = sigma[['Estimate', 'Standard Error', 'Lower Bound', 'Upper Bound', 'Sex']]
                    else:
                        sigma.rename(columns={'estimate': 'Estimate', 'se': 'Standard Error', 'lwr': 'Lower Bound', 'upr': 'Upper Bound'}, inplace=True)
                        sigma = sigma[['Estimate', 'Standard Error', 'Lower Bound', 'Upper Bound']]
                    sigma = sigma.replace([np.inf, -np.inf], 'Inf')
                    sigma = sigma.replace([np.nan], 'NA')
                    sigma = sigma.to_dict(orient='records')

                summary = pd.DataFrame(scr_results.rx2('summary'))
                summary.rename(columns={'best_model': 'Best Model', 'best_model_formula': 'Best Model Formula'}, inplace=True)
                summary = summary.replace([np.inf, -np.inf], 'Inf')
                summary = summary.replace([np.nan], 'NA')
                summary = summary.to_dict(orient='records')

                aic = pd.DataFrame(scr_results.rx2('aic'))
                aic = aic.replace([np.inf, -np.inf], 'Inf')
                aic = aic.replace([np.nan], 'NA')
                aic = aic.to_dict(orient='records')

                cr = pd.DataFrame(scr_results.rx2('cr'))
                cr = cr.replace([np.inf, -np.inf], 'Inf')
                cr = cr.replace([np.nan], 'NA')
                cr = cr.to_dict(orient='records')

                message = str(scr_results.rx2('message')[0])

                raster_df = pd.DataFrame(scr_results.rx2('raster_df'))
                if raster_df.empty:
                    raster_df = []
                else:
                    raster_df['lat'] = raster_df.apply(lambda row: utm.to_latlon(row['X']*1000, row['Y']*1000, zone_number, grid_zone)[0], axis=1)
                    raster_df['lng'] = raster_df.apply(lambda row: utm.to_latlon(row['X']*1000, row['Y']*1000, zone_number, grid_zone)[1], axis=1)
                    raster_df = raster_df[['lat', 'lng', 'density']]
                    max_density = raster_df['density'].max()
                    raster_df = raster_df.to_dict(orient='records')
                    raster_df.append({'max_density': str(max_density)})

                sites_density = pd.DataFrame(scr_results.rx2('sites_density'))
                sites_density = sites_density.to_dict(orient='records')

                # Upload the files to S3
                scr_files = []
                for i in range(len(plots)):
                    fileName = user_folder+'/docs/' + 'SCR_' + plots[i] + '_' + datetime.now().strftime('%Y-%m-%d_%H:%M:%S') + '.JPG'
                    temp_file = open(temp_files[i].name, 'rb')
                    s3client.put_object(Bucket=bucket,Key=fileName,Body=temp_file)
                    scr_files.append("https://"+ bucket + ".s3.amazonaws.com/" + fileName)
                    temp_files[i].close()

                    # Schedule deletion
                    # deleteFile.apply_async(kwargs={'fileName': fileName}, countdown=21600)


                results = {
                    'scr_files': scr_files,
                    'density': density,
                    'abundance': abundance,
                    'det_prob': det_prob,
                    'sigma': sigma,
                    'summary': summary,
                    'aic': aic,
                    'cr': cr,
                    'message': message,
                    'individual_counts': individual_counts,
                    'raster': raster_df,
                    'sites_density': sites_density
                }

                status = 'SUCCESS'
                error = None

    except Exception as exc:
        print(' ')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(traceback.format_exc())
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(' ')
        status = 'FAILURE'
        error = str(exc)

    finally:
        db.session.remove()
        for temp_file in temp_files:
            temp_file.close()

    return { 'status': status, 'error': error, 'scr_results': results }
