from utils import json_handler as json

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    json_file_url = "C:/Users/user/Documents/UNIPI/python/garmin-data/DI_CONNECT/DI-Connect-Fitness/teogoulas@gmail" \
                    ".com_0_summarizedActivities.json "
    json_string = json.import_json(json_file_url)
    json.beatify_json(json_string, "C:/Users/user/Documents/UNIPI/python/garmin-data/formatted",
                      "summarized_activities.json")
