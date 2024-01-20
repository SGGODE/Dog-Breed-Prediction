import os
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

def load_model(model_path):
    """
    Loads a saved model from a specified path.
    """
    print(f"Loadin saved model from{model_path}")
    model = tf.keras.models.load_model(model_path,
                                       custom_objects={"KerasLayer":hub.KerasLayer})
    return model


model = load_model('model.h5')  # Load your trained deep learning model

unique_breed = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier']

@app.route('/')
def home():
    return render_template('index.html')

BATCH_SIZE = 32
def batch_maker(X,y=None,batch_size=BATCH_SIZE,validation=False,test=False):
    """
    so lets do it from the training and validation batches
    """
    if test:
       print("Creating the test data batches.....")
       data=tf.data.Dataset.from_tensor_slices((tf.constant(X)))
       data_batch = data.map(process_image).batch(batch_size)
       return data_batch
    elif validation:
       print("Creating the validation data batches.....")
       data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
       data_batch = data.map(get_image_labels).batch(batch_size)
       return data_batch
    else:
       print("Creating the training data batches.....")
       data=tf.data.Dataset.from_tensor_slices((tf.constant(X),tf.constant(y)))
       data = data.shuffle(buffer_size=len(X))
       data_batch = data.map(get_image_labels).batch(batch_size)
       return data_batch

def get_prediction_label(prediction_proba):
     """
      turn an array of prediction probabilities into a label.
     """
     return unique_breed[np.argmax(prediction_proba)]

Img_size=224

def process_image(image_path,img_size=Img_size):
    """
    take an image file path and turns the image into a tensor
    """
    image = tf.io.read_file(image_path) #read the image file
    image = tf.image.decode_jpeg(image,channels=3) #(RGB) to 0 to 255 array
    image = tf.image.convert_image_dtype(image,tf.float32)#0 to 255 to 0 to 1
    image = tf.image.resize(image,size=[Img_size,Img_size])

    return image

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the uploaded image file
        img_file = request.files['image']
        
        # Save the image to the 'static/images' directory
        img_path = f'C:/Users/HP/OneDrive/Desktop/New folder/first_step/images/{img_file.filename}'
        img_file.save(img_path)
        custom_path = "C:/Users/HP/OneDrive/Desktop/New folder/first_step/images/"
        custom_filenames = [custom_path + fname for fname in os.listdir(custom_path)]
# for it in custom_filenames:
#     print(it)
        custom_test = batch_maker(X=custom_filenames,test=True)
        # Preprocess the image
        # img = image.load_img(img_path, target_size=(224, 224))
        # img_array = batch_maker(X=img_path,test=True)
        #print(img_array.shape)
        #img_array /= 255.0  # Normalize the image

        # Make predictions
        predictions = model.predict(custom_test,verbose=1)
        # You might need post-processing depending on your model's output format

        # Get the predicted breed (replace with your logic)
        predicted_breed = get_prediction_label(predictions[0])

        return render_template('index.html', prediction=predicted_breed, img_path=img_path)

if __name__ == '__main__':
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()
