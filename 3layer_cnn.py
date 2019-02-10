# coding=utf-8
'''
Created on 2019\1\23

@author: ZJ_Xu
'''
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def ImageDataGenerator_sample():
    datagen = ImageDataGenerator(
        rotation_range=40,    # 随机图片角度0-180
        width_shift_range=0.2,  # 随机水平移动0-1
        height_shift_range=0.2, # 随机竖直移动0-1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')    # 需要时进行像素填充

    img = load_img('dataset/train/cat.0.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
        i += 1
        if i > 20:
            break
    train_generator = datagen.flow_from_directory(
        "dataset/tra",  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')

def load_data(train_file, validation_file, test_file):
    print("load data...")
    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,   # rescale值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数。
        shear_range=0.2,    # 剪切变换程度
        zoom_range=0.2,     # 随机放大
        horizontal_flip=True)   # 水平翻转

    # this is the augmentation configuration we will use for testing:
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        train_file,  # this is the target directory
        target_size=(150, 150),  # all images will be resized to 150x150
        batch_size=32,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        validation_file,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    test_generator = test_datagen.flow_from_directory(
        test_file,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
    return train_generator, validation_generator, test_generator

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense

def threelayers_cnn(train_generator, validation_generator, test_generator):

    print('Build model...')
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])



    print('Train...')
    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=20,
            verbose=2,
            validation_data=validation_generator,
            validation_steps=2000)

    score, acc = model.evaluate_generator(generator=test_generator, steps=2000)
    # score, acc = model.evaluate(test_generator)
    print('Test score:', score)
    print('Test accuracy:', acc)


    # model.save_weights('first_try.h5')  # always save your weights after training or during training
    model.save('20190210_2class_10000_2000_500_model.h5')


if __name__=="__main__":
    # ImageDataGenerator_sample()

    train_generator, validation_generator, test_generator = load_data(train_file='./dataset/train', validation_file='./dataset/validation', test_file='./dataset/test')
    threelayers_cnn(train_generator, validation_generator, test_generator)