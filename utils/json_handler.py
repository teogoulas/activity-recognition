import json
import os

from utils import constants


def import_json(dir_path):
    json_object = []
    for entry in os.listdir(dir_path):
        file_path = os.path.join(dir_path, entry)
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


def extract_features(json_object):
    data_set = []
    for obj in json_object:
        for activity in obj[0]['summarizedActivitiesExport']:
            dist = {}
            for feature in constants.RAW_FEATURES:
                if feature in activity:
                    dist[feature] = activity[feature]
            data_set.append(dist)
    return data_set
