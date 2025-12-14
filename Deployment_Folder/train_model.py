import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import json
import os

# --- CONFIGURATION ---
TRAIN_DIR = 'dataset/train' # Ensure this path exists and has subfolders
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def train_engine():
    # 1. Data Pipeline with Augmentation (Pro-level robustness)
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Automatically create validation set if you don't have separate folders
    )

    print("Loading Data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # 2. Save Class Mappings (Crucial for correct frontend display)
    # This ensures 'broken' is mapped to 0, 'healthy' to 1, etc., correctly.
    class_indices = train_generator.class_indices
    with open('classes.json', 'w') as f:
        json.dump(class_indices, f)
    print(f"Class Mappings Saved: {class_indices}")

    # 3. Model Architecture (MobileNetV2 - Optimized for Web/Remote)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze base to prevent destroying learned features
    base_model.trainable = False 

    # Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x) # Prevents overfitting
    predictions = Dense(3, activation='softmax')(x) # 3 Classes: Healthy, Discolored, Broken

    model = Model(inputs=base_model.input, outputs=predictions)

    # 4. Compile and Train
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting Training...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=15, # Adjust based on your hardware
        verbose=1
    )

    # 5. Fine-Tuning (Optional but recommended for "Pro" results)
    print("Fine-tuning model...")
    base_model.trainable = True
    # Freeze first 100 layers, train the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False
        
    model.compile(optimizer=Adam(learning_rate=1e-5), # Lower learning rate
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
                  
    model.fit(train_generator, validation_data=validation_generator, epochs=5)

    # 6. Save Model
    model.save('corn_model.h5')
    print("SUCCESS: Model saved as 'corn_model.h5'")

if __name__ == "__main__":
    if os.path.exists(TRAIN_DIR):
        train_engine()
    else:
        print(f"ERROR: Directory '{TRAIN_DIR}' not found. Please create the dataset structure first.")