import json


def import_json(file_path):
    try:
        with open(
                file_path, encoding="utf8") as f:
            json_string = f.read()
        json_object = json.loads(json_string)
        return json_object

    except Exception as e:
        print(repr(e))


def beatify_json(json_object, file_path, file_name):
    try:
        formatted_json = json.dumps(json_object, indent=4, sort_keys=True)
        with open("{:s}/{:s}".format(file_path, file_name), "w", encoding="utf-8") as f:
            f.write(formatted_json)
    except Exception as e:
        print(repr(e))


def extract_features(json_object, features):
    data_set = []
    for activity in json_object[0]['summarizedActivitiesExport']:
        dist = {}
        for feature in features:
            if feature in activity:
                dist[feature] = activity[feature]
        data_set.append(dist)
    return data_set
