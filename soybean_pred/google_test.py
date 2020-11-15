from PIL import Image
import numpy as np
from google_pred import predict_soybean_class
import os

credential_path = 'lucid-honor-295522-305e6cacd6aa.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

# test image
img = Image.open('guatemala-volcano_00000000_post_disaster.png')
img = img.resize((256,256))
img_array = np.array(img)
print(img_array.shape)
img_array = np.expand_dims(img_array,0)
#print(img_array.shape)
print('Predicted Soybean Class: ' + str(predict_soybean_class(img_array)))
		
