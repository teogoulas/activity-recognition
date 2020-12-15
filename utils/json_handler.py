import json


def import_json(file_path):
    try:
        with open(
                file_path, encoding="utf8") as f:
            json_string = f.read()
        data_set = json.loads(json_string)
        return data_set

    except Exception as e:
        print(repr(e))


def beatify_json(json_object, file_path, file_name):
    try:
        formatted_json = json.dumps(json_object, indent=4, sort_keys=True)
        with open("{:s}/{:s}".format(file_path, file_name), "w", encoding="utf-8") as f:
            f.write(formatted_json)
    except Exception as e:
        print(repr(e))
