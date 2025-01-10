from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generator():
    return ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        rescale=1./255
    )
