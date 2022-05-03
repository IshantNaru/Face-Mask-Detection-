import splitfolders as sf
from keras.preprocessing.image import ImageDataGenerator as IDG


def splittingfolders(path, output):
    try:
        sf.ratio(input=path, output=output, seed=1337, ratio=(0.6, 0.2, 0.2))
    except Exception as e:
        print(e)
    else:
        print("Successfully split the folders")


def imagepreprocess(train, val):
    # Creating Image Data Generator train and test objects
    train_datagen = IDG(rescale=1. / 255,
                        rotation_range=10,
                        # width_shift_range=0.2,
                        # height_shift_range=0.2,
                        # shear_range=0.3,
                        # zoom_range=0.4,
                        horizontal_flip=True)
    test_datagen = IDG(rescale=1. / 255)

    # Creating train and validation generators
    try:
        train_generator = train_datagen.flow_from_directory(train,
                                                            color_mode='rgb',
                                                            target_size=(180, 180),
                                                            batch_size=200,
                                                            classes={'WithoutMask': 0, 'WithMask': 1},
                                                            class_mode='binary',
                                                            seed=45)

        validation_generator = test_datagen.flow_from_directory(val,
                                                                color_mode='rgb',
                                                                target_size=(180, 180),
                                                                batch_size=90,
                                                                classes={'WithoutMask': 0, 'WithMask': 1},
                                                                class_mode='binary',
                                                                seed=45)
        return train_generator, validation_generator

    except Exception as e:
        print("Some error loading the images")
