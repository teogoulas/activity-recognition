from utils import constants


def find_activities(data_set):
    activities = []
    for activity in data_set:
        activity_type = activity["activityType"]
        if activity_type not in activities:
            activities.append(activity_type)
    return activities


def group_activities(data_set):
    activity_groups = {"biking": []}
    for activity_type in constants.FILTERED_FEATURES:
        if activity_type in ["cycling", "mountain_biking", "trail_running"]:
            continue
        activity_groups[activity_type] = []

    for activity in data_set:
        act_type = activity["activityType"]
        if act_type in constants.FILTERED_FEATURES:
            if act_type == "cycling" or act_type == "mountain_biking":
                activity_groups["biking"].append(activity)
            elif act_type == "trail_running":
                activity_groups["running"].append(activity)
            else:
                activity_groups[act_type].append(activity)
    return activity_groups
