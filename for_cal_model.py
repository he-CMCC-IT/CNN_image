# coding=utf-8
'''
Created on 2019\1\23

@author: ZJ_Xu
'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def test(filename):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(filename,
                                       target_size=(150, 150),
                                       batch_size=32,
                                       class_mode='binary')

    from keras.models import load_model
    test_model = load_model('20190210_2class_10000_2000_500_model.h5')


    score, acc = test_model.evaluate(test_generator)
    print('Test score:', score)
    print('Test accuracy:', acc)

    print(test_model.predict_classes(test_generator))
    print(test_model.predict(test_generator))


if __name__=="__main__":
    test('./dataset/test0')