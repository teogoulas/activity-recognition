import json
import os

import pandas as pd
from models.activity import Activity
from utils import constants


def import_json():
    json_object = []
    for entry in os.listdir(constants.RAW_DATA_DIR):
        file_path = os.path.join(constants.RAW_DATA_DIR, entry)
        if os.path.isfile(file_path):
            try:
                with open(
                        file_path, encoding="utf8") as f:
                    json_string = f.read()
                json_object.append(json.loads(json_string))

            except Exception as e:
                print(repr(e))

    return json_object


def beatify_json(json_object, file_path, file_name):
    try:
        formatted_json = json.dumps(json_object, indent=4, sort_keys=True)
        with open("{:s}/{:s}".format(file_path, file_name), "w", encoding="utf-8") as f:
            f.write(formatted_json)
    except Exception as e:
        print(repr(e))


def extract_attribute(activity, key):
    return activity[key] if key in activity.keys() else None


def extract_features(json_object):
    data_set = []
    for obj in json_object:
        for activity in obj[0]['summarizedActivitiesExport']:
            activity_type = extract_attribute(activity, 'activityType')
            if activity_type in constants.FILTERED_FEATURES:
                if activity_type == "cycling" or activity_type == "mountain_biking":
                    activity_type = "biking"
                elif activity_type == "trail_running":
                    activity_type = "running"
                elif activity_type == "other":
                    activity_type = "indoor_cardio"
                new_activity = Activity(activity_type=activity_type,
                                        aerobic_training_effect=extract_attribute(activity, 'aerobicTrainingEffect'),
                                        anaerobic_training_effect=extract_attribute(activity, 'anaerobicTrainingEffect'),
                                        avg_hr=extract_attribute(activity, 'avgHr'),
                                        avg_speed=extract_attribute(activity, 'avgSpeed'),
                                        calories=extract_attribute(activity, 'calories'),
                                        distance=extract_attribute(activity, 'distance'),
                                        # max_double_cadence=extract_attribute(activity, 'maxDoubleCadence'),
                                        max_ftp=extract_attribute(activity, 'maxFtp'),
                                        max_hr=extract_attribute(activity, 'maxHr'),
                                        duration=extract_attribute(activity, 'duration'))
                data_set.append(new_activity)
    return pd.DataFrame([act.as_dict() for act in data_set])
