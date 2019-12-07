import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time

start = time.time()


#os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#Define Path


model_path = '/home/toto/CNN/models/weight.best.hdf5'
model_weights_path = '/home/toto/CNN/models/weight.best.hdf5'
test_path = '/home/toto/slice/Data200Pixels/testing/malignant'	
            

#Load the pre-trained models

model = load_model(model_path)
model.load_weights(model_weights_path)



#Define image parameters

img_width, img_height = 224,224

count_benign = 0
count_malignant = 0

 
#Prediction Function

def predict(file):

  x = load_img(file, target_size=(img_width,img_height))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = model.predict(x)
  result = array[0]

  print(result)

  answer = np.argmax(result)

  if answer == 0:

    print("Predicted: benign")
    

  elif answer == 1:

    print("Predicted: malignant")    
  

  return answer


#Walk the directory for every image

for i, ret in enumerate(os.walk(test_path)):

  for i, filename in enumerate(ret[2]):

    if filename.startswith("."):

      continue

    

    print(ret[0] + '/' + filename)

    result = predict(ret[0] + '/' + filename)
    print(result)
    if result == 0:
       count_benign = count_benign+1
    elif result ==1:
       count_malignant = count_malignant+1
    

    print(count_benign)
    print(count_malignant) 
        
    print(" ")



#Calculate execution time

end = time.time()

dur = end-start



if dur<60:

    print("Execution Time:",dur,"seconds")

elif dur>60 and dur<3600:

    dur=dur/60

    print("Execution Time:",dur,"minutes")

else:

    dur=dur/(60*60)

    print("Execution Time:",dur,"hours")


