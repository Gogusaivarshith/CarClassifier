import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import shutil


car_train_dir = r'C:\Users\varsh\Videos\combined_dataset\train\cars'  
car_validation_dir = r'C:\Users\varsh\Videos\combined_dataset\test\cars' 
non_car_train_dir = r'C:\Users\varsh\Videos\combined_dataset\train\non-cars'  
non_car_validation_dir = r'C:\Users\varsh\Videos\combined_dataset\test\non-cars'  


os.makedirs('combined_dataset/train/cars', exist_ok=True)
os.makedirs('combined_dataset/train/non-cars', exist_ok=True)
os.makedirs('combined_dataset/test/cars', exist_ok=True)
os.makedirs('combined_dataset/test/non-cars', exist_ok=True)


def move_images(source_dir, target_dir):
    for root, _, files in os.walk(source_dir):
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)
            if not os.path.exists(target_file):  
                shutil.move(source_file, target_dir)


move_images(car_train_dir, 'combined_dataset/train/cars')
move_images(car_validation_dir, 'combined_dataset/test/cars')


move_images(non_car_train_dir, 'combined_dataset/train/non-cars')
move_images(non_car_validation_dir, 'combined_dataset/test/non-cars')


train_dir = 'combined_dataset/train'
validation_dir = 'combined_dataset/test'


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.fit(train_generator, epochs=10, validation_data=validation_generator)


model.save('car_classifier_model.h5')


def predict_car(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(64, 64))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "Non-car" if prediction[0][0] >= 0.5 else "car"


test_image_path = r"C:\Users\varsh\OneDrive\Desktop\download.jpg"
result = predict_car(test_image_path)
print(f"The prediction for the test image is: {result}")
