from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from keras.models import load_model
import keras.backend as k

app = Flask(__name__)
model = load_model('model_70_15_15_b256_v5.h5')
classes = ["AluCan", "Glass", "PET"]

@app.route('/api/predict/', methods=['GET', 'POST'])
def predict():
    if request.files.get('image'):
        data = request.files['image'].read()
        # img = Image.open(data).convert('RGB')
        img_tf = tf.io.decode_image(data, channels=3)
        re_img = tf.image.resize(img_tf, [224, 224])
        # re_img = img.resize((224,224))
        nparr = np.true_divide(re_img, 255)
        nparr = nparr.reshape(1, 224, 224, 3)
        # image = tf.cast(tf.expand_dims(nparr, axis=0), tf.int16)
        # print(nparr)
        # instances_list = tf.cast(nparr, tf.float16).numpy().tolist()
        # image.numpy().tolist()
        print(nparr.shape)
        prediction = model.predict(nparr)
        predict_class = classes[tf.argmax(prediction[0])]
        k.clear_session()
        return jsonify(predict_class)

if __name__ == '__main__':
    app.run()
