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

'''Cleans up the local worker by stopping its task consumption before re-queueing its active and reserved tasks.'''

from worker import app as celery
from config import Config
import importlib
import re
import sys

celery.control.cancel_consumer(queue=Config.QUEUE, destination=[Config.WORKER_NAME])
inspector = celery.control.inspect()

active_tasks = []
active_tasks.extend(inspector.active()[Config.WORKER_NAME])
active_tasks.extend(inspector.reserved()[Config.WORKER_NAME])

for active_task in active_tasks:
    task_name = active_task['name']
    if not task_name.startswith('WorkR.'):
        continue
    module = importlib.import_module('worker')
    function_name = task_name.split('.', 1)[1]
    active_function = getattr(module, function_name)
    kwargs = active_task['kwargs']
    task_id = active_task['id']
    priority = active_task['delivery_info']['priority']
    print('Rescheduling {} with args {}'.format(task_name, kwargs))
    active_function.apply_async(kwargs=kwargs, queue=Config.QUEUE, priority=priority, task_id=task_id)

sys.exit(99)