#%%
#Import Packages
import os
import datetime
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers,optimizers,losses,callbacks,metrics,applications

#%%
#1. Data Loading
file_path = r"C:\Users\USER\Desktop\STEP_AI01_Muhammad_Iqmal_Hakim_Assesment_3\Concrete Crack Image Datasets"
dir_path = pathlib.Path(file_path)

#%%
#2. Data Preparation
#(A) Define batch_size & image_size
SEED = 64
IMG_SIZE = (180,180)

#(B) Load the data into tensorflow dataset using the specific method
train_dataset = tf.keras.utils.image_dataset_from_directory(dir_path, shuffle=True, validation_split=0.2, subset="training",seed=SEED, image_size = IMG_SIZE , batch_size=10)

val_dataset = tf.keras.utils.image_dataset_from_directory(dir_path, shuffle=True, validation_split=0.2, subset="validation",seed=SEED, image_size = IMG_SIZE , batch_size=10)

#%%
#3. Display images as example
class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
for images,labels in train_dataset.take(1):
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')

#%%
#4. Further split the validation dataset into validation-test split
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#%%
#5. Convert the BatchDataset into PrefetchDataset
AUTOTUNE = tf.data.AUTOTUNE
pf_train = train_dataset.prefetch(buffer_size=AUTOTUNE)
pf_val = validation_dataset.prefetch(buffer_size=AUTOTUNE)
pf_test = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
#6. Create a pipeline for data augmentation
data_augmentation = keras.Sequential()
data_augmentation.add(layers.RandomFlip('horizontal'))
data_augmentation.add(layers.RandomRotation(0.2))

#%%
# Apply data augmentation to test
for images,labels in pf_train.take(1):
    first_image = images[0]
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        augmented_image = data_augmentation(tf.expand_dims(first_image,axis=0))
        plt.imshow(augmented_image[0]/255.0)
        plt.axis('off')

#%%
#7. Prepare the layer for data preprocessing

preprocess_input = applications.mobilenet_v2.preprocess_input

#8. Apply transfer learning
IMG_SHAPE = IMG_SIZE + (3,)
feature_extractor = applications.MobileNetV2(input_shape=IMG_SHAPE,include_top=False,weights='imagenet')

#Disable the training for the feature extractor(freeze the layers)

feature_extractor.trainable = False
feature_extractor.summary()
keras.utils.plot_model(feature_extractor,show_shapes=True)

# %%
#9. Create the classification layers
global_avg = layers.GlobalAveragePooling2D()
output_layer = layers.Dense(len(class_names),activation = 'softmax')

# %%
#10. Use functional API to link all of the modules together
inputs = keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = feature_extractor(x)
x = global_avg(x)
x = layers.Dropout(0.3)(x)
outputs = output_layer(x)

model = keras.Model(inputs=inputs,outputs=outputs)

#To show model architecture
keras.utils.plot_model(model,show_shapes=True)
model.summary()

#%%
#11. Compile the model
optimizer = optimizers.Adam(learning_rate=0.0001)
loss = losses.SparseCategoricalCrossentropy()
model.compile(optimizer=optimizer,loss=loss,metrics=['accuracy'])

#Evaluate the model before model training
loss0,accuracy0 = model.evaluate(pf_val)
print("Loss = ",loss0)
print("Accuracy = ",accuracy0)

#%%
#Tensorboard & Earlystopping callbacks
log_path = os.path.join('log_dir','tl_demo',datetime.datetime.now().strftime('%Y%m%d - %H%M%S'))
tb = callbacks.TensorBoard(log_dir = log_path)
es = keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)

#%%
#%%
#Train Model
EPOCHS = 5
BATCH_SIZE = 64
history = model.fit(pf_train,validation_data=pf_val,epochs=EPOCHS,batch_size = BATCH_SIZE, callbacks=[tb,es])

#%%
#12. Evaluate the final model
test_loss,test_acc = model.evaluate(pf_test)

print("Loss = ", test_loss)
print("Accuracy = ", test_acc)

#%%
#13. Model Deployment

#Deploy the model using the test data
image_batch, label_batch = pf_test.as_numpy_iterator().next()
predicitons = np.argmax(model.predict(image_batch),axis=1)

#%%
#Compare label and predicition
label_vs_predicition = np.transpose(np.vstack((label_batch,predicitons)))

#%%

plt.figure(figsize=(20,20))

for i in range(len(image_batch)):
    plt.subplot(6,3,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    plt.title(f"Label : {class_names[label_batch[i]]}, Prediction: {class_names[predicitons[i]]}")
    plt.axis('off')
plt.show()

#%%
#%% Model Saving
#save trained tf model
model.save('model.h5')

