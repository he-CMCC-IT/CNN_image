# coding=utf-8
'''
Created on 2019\1\23

@author: ZJ_Xu
'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import numpy as np
def see_reslult(path, test_model):
    # test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_generator = test_datagen.flow_from_directory(filename,
    #                                    target_size=(150, 150),
    #                                    batch_size=32,
    #                                    class_mode='binary')

    #load the image
    image = cv2.imread(path)
    # orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (800, 180))  # 宽*高
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)




    # score, acc = test_model.evaluate(test_generator)
    # print('Test score:', score)
    # print('Test accuracy:', acc)

    print(test_model.predict_classes(image))
    print(test_model.predict(image))


if __name__=="__main__":
    from keras.models import load_model
    test_model = load_model('20190210_2class_10000_2000_500_model.h5')

    sent = input('image name:')
    while sent != 'exit':
        file_path = './dataset/train/dogs/'+ sent +'.jpg'
        see_reslult(file_path, test_model)
        sent = input('image name:')