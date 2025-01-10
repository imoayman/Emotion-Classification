from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generator():
    return ImageDataGenerator(
        rotation_range=40,
        zoom_range=0.2,                 
        width_shift_range=0.3,         
        height_shift_range=0.3,         
        shear_range=0.2,                
        horizontal_flip=True,         
        brightness_range=[0.7, 1.3], 
        rescale=1./255                
    )
