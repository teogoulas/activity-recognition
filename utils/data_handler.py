import json
import os

import pandas as pd
from sklearn import preprocessing
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
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


def data_preprocessing(data_set):
    data_set = data_visualization(data_set)
    print("After data preprocessing the following features were selected:")
    for col in data_set:
        if col == 'activity_type':
            continue
        print(f"- {col}")

    # remove NaN values and convert to np array
    filtered_data = data_set.dropna()
    print(f"Dataset size: {len(filtered_data)} samples")

    y = filtered_data['activity_type'].to_numpy()
    original = filtered_data.iloc[:, 1:11].to_numpy()

    # Minmax scaler
    mm_scaler = preprocessing.MinMaxScaler()
    minmax_scaler = mm_scaler.fit_transform(original)

    # Robust scaler
    rs = preprocessing.RobustScaler()
    robust_scaler = rs.fit_transform(original)
    hybrid_scaler = rs.fit_transform(minmax_scaler)

    # Standard scaler
    ss = preprocessing.StandardScaler()
    standard_scaler = ss.fit_transform(original)
    hybrid_scaler = ss.fit_transform(hybrid_scaler)

    # Normalizer
    normalizer = preprocessing.Normalizer()
    x_normed = normalizer.fit_transform(original)
    hybrid_scaler = normalizer.fit_transform(hybrid_scaler)

    return {"original": original, "minmax_scaler": minmax_scaler, "robust_scaler": robust_scaler,
            "normalizer": x_normed,
            "standard_scaler": standard_scaler, "hybrid_scaler": hybrid_scaler, "y": y}


def data_visualization(data_set):
    fig_check_null = px.imshow(data_set.isnull(), labels=dict(x='Features', y="Samples"),
                               x=list(data_set.columns.values), contrast_rescaling='minmax')
    fig_check_null.update_layout(title={
        'text': "NaN values detection",
        'y': 0.95,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }, showlegend=False)
    fig_check_null.show()

    # removed features with multiple NaN values
    data_set = data_set.drop(['elevation_gain', 'elevation_loss', 'max_run_cadence', 'steps', 'avg_stride_length',
                              'min_elevation', 'max_elevation', 'avg_double_cadence', 'max_double_cadence',
                              'max_vertical_speed', 'water_estimated', 'min_respiration_rate', 'max_respiration_rate',
                              'avg_respiration_rate', 'activity_training_load'], axis=1)

    corr = data_set.corr()
    trace = go.Heatmap(z=corr.values,
                       x=corr.index.values,
                       y=corr.columns.values)
    data = [trace]
    fig_corr = go.Figure(data=data, layout={
        'title': {
            'text': "Correlation Matrix",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
    })
    fig_corr.show()

    # removed features with low correlation with activity type
    data_set = data_set.drop(['event_type_id', 'moving_duration'], axis=1)

    # removed features highly correlated with others
    data_set = data_set.drop(['aerobic_training_effect', 'avg_hr', 'start_time_local', 'start_time_gmt',
                              'elapsed_duration', 'max_temperature'], axis=1)

    fig_box = make_subplots(rows=1, cols=len(data_set.columns.values))
    col_count = 1
    for col in data_set:
        fig_box.add_trace(go.Box(y=data_set[col].values, name=data_set[col].name), row=1, col=col_count)
        col_count += 1

    fig_box.update_layout(title={
        'text': "Outliers Detection",
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'
    }, showlegend=False)
    fig_box.show()
    return data_set


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
                                        anaerobic_training_effect=extract_attribute(activity,
                                                                                    'anaerobicTrainingEffect'),
                                        avg_hr=extract_attribute(activity, 'avgHr'),
                                        avg_speed=extract_attribute(activity, 'avgSpeed'),
                                        calories=extract_attribute(activity, 'calories'),
                                        distance=extract_attribute(activity, 'distance'),
                                        max_ftp=extract_attribute(activity, 'maxFtp'),
                                        max_hr=extract_attribute(activity, 'maxHr'),
                                        duration=extract_attribute(activity, 'duration'),
                                        begin_timestamp=extract_attribute(activity, 'beginTimestamp'),
                                        event_type_id=extract_attribute(activity, 'eventTypeId'),
                                        start_time_gmt=extract_attribute(activity, 'startTimeGmt'),
                                        start_time_local=extract_attribute(activity, 'startTimeLocal'),
                                        elevation_gain=extract_attribute(activity, 'elevationGain'),
                                        elevation_loss=extract_attribute(activity, 'elevationLoss'),
                                        max_run_cadence=extract_attribute(activity, 'maxRunCadence'),
                                        steps=extract_attribute(activity, 'steps'),
                                        avg_stride_length=extract_attribute(activity, 'avgStrideLength'),
                                        avg_fractional_cadence=extract_attribute(activity, 'avgFractionalCadence'),
                                        max_fractional_cadence=extract_attribute(activity, 'maxFractionalCadence'),
                                        elapsed_duration=extract_attribute(activity, 'elapsedDuration'),
                                        moving_duration=extract_attribute(activity, 'movingDuration'),
                                        min_temperature=extract_attribute(activity, 'minTemperature'),
                                        max_temperature=extract_attribute(activity, 'maxTemperature'),
                                        min_elevation=extract_attribute(activity, 'minElevation'),
                                        max_elevation=extract_attribute(activity, 'maxElevation'),
                                        avg_double_cadence=extract_attribute(activity, 'avgDoubleCadence'),
                                        max_double_cadence=extract_attribute(activity, 'maxDoubleCadence'),
                                        max_vertical_speed=extract_attribute(activity, 'maxVerticalSpeed'),
                                        lap_count=extract_attribute(activity, 'lapCount'),
                                        water_estimated=extract_attribute(activity, 'waterEstimated'),
                                        min_respiration_rate=extract_attribute(activity, 'minRespirationRate'),
                                        max_respiration_rate=extract_attribute(activity, 'maxRespirationRate'),
                                        avg_respiration_rate=extract_attribute(activity, 'avgRespirationRate'),
                                        activity_training_load=extract_attribute(activity, 'activityTrainingLoad'))
                data_set.append(new_activity)

    return data_preprocessing(pd.DataFrame([act.as_dict() for act in data_set]))
