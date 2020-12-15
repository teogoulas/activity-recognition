from utils import json_handler as json

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    features = ["activityType", "aerobicTrainingEffect", "anaerobicTrainingEffect", "avgHr", "avgSpeed", "calories",
                "distance", "maxDoubleCadence", "maxFtp", "maxHr", "maxSpeed"]
    json_file_url = "C:/Users/user/Documents/UNIPI/python/garmin-data/DI_CONNECT/DI-Connect-Fitness/teogoulas@gmail" \
                    ".com_0_summarizedActivities.json "
    json_object = json.import_json(json_file_url)
    data_set = json.extract_features(json_object, features)
