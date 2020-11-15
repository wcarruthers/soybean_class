import googleapiclient.discovery
#from google.api_core.client_options import ClientOptions #only use if testing locally
import numpy as np
#import os

#print("Importing being called")

#credential_path = 'lucid-honor-295522-305e6cacd6aa.json'
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

def predict_json(project, region, model, instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the Cloud ML Engine Model is deployed.
        region (str): regional endpoint to use; set to None for ml.googleapis.com
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    #print("I am being called json")
    # Create the ML Engine service object.
    # To authenticate set the environment variable
    # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
    prefix = "{}-ml".format(region) if region else "ml"
    api_endpoint = "https://{}.googleapis.com".format(prefix)
    client_options = ClientOptions(api_endpoint=api_endpoint)
    service = googleapiclient.discovery.build(
        'ml', 'v1', client_options=client_options)
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']
    
def predict_soybean_class(img_array):
    class_names = ['broadleaf', 'grass', 'soil',  'soybean']
    PROJECT = 'lucid-honor-295522'
    REGION = 'us-central1'
    MODEL = 'soybean_class'
    INSTANCES = img_array.tolist()
    score = predict_json(PROJECT, REGION, MODEL, INSTANCES)
    return np.array(class_names)[np.argmax(score)]
  