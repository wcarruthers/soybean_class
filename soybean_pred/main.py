from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from google_pred import predict_uva_landmark

app = Flask(__name__, template_folder='templates')


@app.route('/')
def upload_file():
    return render_template('index.html')

@app.route('/uploader', methods = ['POST'])
def upload_image_file():
    if request.method == 'POST':
        img = Image.open(request.files['file'].stream)
        img = img.resize((256,256))
        img_array = np.array(img)
        print(img_array.shape)
        img_array = np.expand_dims(img_array,0)
        return 'Predicted Landmark: ' + str(predict_uva_landmark(img_array))
		
if __name__ == '__main__': 
  
    # run() method of Flask class runs the application  
    # on the local development server. 
    app.run()
