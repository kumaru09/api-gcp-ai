from PIL import Image
from flask import Flask, request, jsonify
import numpy as np
import googleapiclient.discovery
from google.api_core.client_options import ClientOptions
import os
import tensorflow as tf

app = Flask(__name__)

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceAccount.json'

classes = ["AluCan", "Glass", "PET"]

@app.route('/api/predict/', methods=['GET', 'POST'])
def predict():
    if request.files.get('image'):
        data = request.files['image']
        img = Image.open(data).convert('RGB')
        #img_tf = tf.io.decode_image(data, channels=3)
        #re_img = tf.image.resize(img_tf, [224, 224])
        re_img = img.resize((224,224))
        nparr = np.true_divide(re_img, 255)
        image = tf.cast(tf.expand_dims(nparr, axis=0), tf.int16)
        instances_list = image.numpy().tolist()
        print(image.shape)
        prediction = predict_json('instant-matter-331109','asia-southeast1','bnn_ai_model',instances_list,'test')
        predict_class = classes[tf.argmax(prediction[0])]
        return jsonify(predict_class)

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

if __name__ == '__main__':
    app.run(debug=True)
