from utils import json_handler as json
from utils import utilities as utils

if __name__ == '__main__':
    json_files_url = "raw_data/"
    json_object = json.import_json(json_files_url)
    data_set = json.extract_features(json_object)
    groups = utils.group_activities(data_set)
    print("end")
