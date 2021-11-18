import numpy as np 
import os
import sys

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask , render_template , request , url_for

from werkzeug.utils import secure_filename

app = Flask(__name__)

Model_path = "InceptionV3.h5"

model = load_model(Model_path)

def model_predict(img_path,model):
    print(img_path)
    img= image.load_img(img_path,target_size=(224,224))

    img_array= image.img_to_array(img)

    img_array=img_array/255

    img_array=np.expand_dims(img_array,axis=0)

    preds = model.predict(img_array)
    preds1 = np.argmax(preds)
    c=""
    if(preds1==0):
       c= "This is diseased cotton leaf"

    elif(preds1==1):
        c="This is diseased cotton plant"

    elif(preds1==2):
        c="This is fresh cotton leaf"

    elif(preds1==3):
        c="This is fresh cotton plant"



    return(c)



@app.route("/",methods=['GET'])
def index():
    return(render_template('index.html'))


@app.route("/predict",methods=['GET','POST'])
def upload():
    if(request.method=="POST"):

        f = request.files['file']

        base_path = os.path.dirname(__file__)

        file_path = os.path.join(base_path,'uploads',secure_filename(f.filename))

        f.save(file_path)


        prediction=model_predict(file_path,model)
        result=prediction
        return(result)

    return(None)


if __name__ == '__main__':
    app.run(port=5001,debug=True)