from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt

model = load_model('devanagari.h5')

recognize_image = cv2.imread('character.jpg')

char_list = ['क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त', 'थ', 'द', 'ध', 'न', 'प',
             'फ', 'ब', 'भ', 'म', 'य', 'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'क्ष', 'त्र', 'ज्ञ', '१', '२', '३', '४', '५',
             '६', '७', '८', '९']


def adjust_gamma(img, gamma=2.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(img, table)


imgrgb = cv2.cvtColor(recognize_image, cv2.COLOR_BGR2GRAY)
imgrgb = cv2.bitwise_not(imgrgb)
imgrgb = cv2.fastNlMeansDenoising(imgrgb, None, 7, 21, 5)
# imgrgb = cv2.bilateralFilter(imgrgb,10,0,100)
kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
im = cv2.filter2D(imgrgb, -1, kernel)
_, im = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)
im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_AREA)
im = adjust_gamma(im)
predict = image.img_to_array(im)
predict = np.expand_dims(predict, axis=0)
plt.imshow(im, cmap='gray')
plt.show()
print(char_list[model.predict_classes(predict)[0]])
