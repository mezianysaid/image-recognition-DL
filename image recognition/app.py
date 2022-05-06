import tensorflow as tf 
import tensorflow.keras as keras
import numpy as np
from flask import Flask, render_template, request
model_path='./static/saved_model/train.ckpt'
restore_model=keras.models.load_model(model_path)
# restore_model.summary()
app=Flask(__name__)
@app.route('/',methods=['GET'])
def index():    
        return render_template('imagerecognition.html');
@app.route('/',methods=['POST'])
def ImageRecognition():
        
        img=request.files['image'];
        image_path='./static/images/'+ img.filename;
        img.save(image_path);
        label=Predict(image_path)
        return render_template('imagerecognition.html',imgs=image_path,l=label);

def Predict(path_image):
          image=keras.preprocessing.image.load_img(path_image,target_size=(180,180))
          img_array=keras.preprocessing.image.img_to_array(image) 
          img_array=tf.expand_dims(img_array,0)
          prediction=restore_model.predict(img_array)
 
          rslt=tf.nn.softmax(prediction[0])
          res=np.argmax(rslt)
          if res == 0:
                  label='Cat'
          if res == 1:
                  label='Dog'
          return label
  
  
if __name__ =='__main__':
        app.run()