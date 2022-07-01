import pandas as pd
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Lambda, Conv2D, MaxPool2D
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

data = pd.read_pickle("E:/veriseti/data8.pickle") #veriseti yüklenmesi
data.keys()

plt.imshow(data["x_train"][12].reshape(32, 32), cmap='gray')
plt.show()

data["x_train"][12]
data["x_train"][12].shape

label_encode = data["y_train"][12] #veri etiketi
data["labels"][label_encode]

np.unique(data["y_train"]) #bütün veri etiketleri (43 adet trafik işareti)

# eğitim verileri boyutları
print("x train shape:", data["x_train"].shape)
print("y train shape:", data["y_train"].shape)
# test verileri boyutları
print("x test shape:", data["x_test"].shape)
print("y test shape:", data["y_test"].shape)
# doğrulama verileri boyutları
print("x validation shape:", data["x_validation"].shape)
print("y validation shape:", data["y_validation"].shape)

x_train = data["x_train"]
x_test = data["x_test"]
x_val = data["x_validation"]
y_train = data["y_train"]
y_val = data["y_validation"]

# reshape işlemleri
x_train = x_train.swapaxes(1,2)
x_train.shape

x_train = x_train.swapaxes(2,3)
x_train.shape

x_val = x_val.swapaxes(1,2)
x_val = x_val.swapaxes(2,3)
print("x val shape:", x_val.shape)

# resimlerin 4'e 4 sıralı ve gri tonlamalı bir şekilde gösterimi
plt.figure(figsize=(10,10))

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(x_train[i], cmap = "gray")
    plt.axis("off")

def resize(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage, 32, 32, 1))
    for i in range(numberOfImage):
        new_array[i] = tf.image.resize(img[i], (32, 32))
    return new_array

x_train_resized = resize(x_train)
x_val_resized = resize(x_val)
print("x validation resized shape:", x_val_resized.shape)

numberOfClass = 43
y_train = to_categorical(y_train, num_classes = numberOfClass)
y_val = to_categorical(y_val, num_classes = numberOfClass)

# modelin hazırlanması
model = Sequential()
model.add(Conv2D(filters = 128, kernel_size = (4, 4), padding = "Same", activation = "relu", input_shape = (32, 32, 1)))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 64, kernel_size = (4, 4), padding = "Same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Conv2D(filters = 32, kernel_size = (4, 4), padding = "Same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(units = 512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(units = numberOfClass, activation = "softmax"))

model.summary()
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

# modelin eğitilmesi
hist = model.fit(x_train_resized, y_train, batch_size = 128, epochs = 10, validation_data = (x_val_resized, y_val))

# accuracy ve loss değerlerinin kontrolü
plt.style.use('seaborn')
plt.figure(figsize = (6, 6))
plt.plot(hist.history['loss'], color = 'b', label = 'Training loss')
plt.plot(hist.history['val_loss'], color = 'r', label = 'Validation loss')
plt.legend()
plt.show()

model.savae('C:/archive/model.h5') # modelin kayıt işlemi
test_model = tf.keras.models.load_model('C:/archvie/model.h5') # kayıtlı modelin yüklenmesi
img = load_img('C:/archive/pass.jpg', color_mode = 'grayscale', target_size = (32, 32, 1)) # test edilecek resmin yüklenmesi

x = img_to_array(img)
x = np.expand_dims(x, axis = 0)
preds = test_model.predict_step(x)
print(preds) # sonuç