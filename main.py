from predictors.predictors import logistic_regression
from utils import json_handler as json
from utils import utilities as utils
import plotly.express as px

if __name__ == '__main__':
    json_object = json.import_json()
    data_set = json.extract_features(json_object)
    logistic_regression(data_set)
    fig = px.scatter_matrix(data_set,
                            dimensions=['aerobic_training_effect', 'anaerobic_training_effect', 'avg_hr', 'calories',
                                        'distance'],
                            color="activity_type")
    fig.write_html("exports/matrix_plot.html")
    fig.show()
    print("end")
