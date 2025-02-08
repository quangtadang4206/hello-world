import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
# Thêm chiều kênh (channel dimension)
x_train = x_train[..., np.newaxis]  # Thêm chiều channel, kết quả (60000, 28, 28, 1)
x_test = x_test[..., np.newaxis]
# Xây dựng mô hình CNN với các cải tiến
model_cnn = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),  # Lớp tích chập
    MaxPooling2D((2, 2)),  # Lớp pooling
    Dropout(0.25),  # Dropout để giảm overfitting
    Conv2D(64, (3, 3), activation='relu'),  # Lớp tích chập thứ hai
    MaxPooling2D((2, 2)),  # Lớp pooling thứ hai
    Dropout(0.25),  # Dropout thêm
    Flatten(),
    Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),  # L2 Regularization
    Dropout(0.5),  # Dropout trước lớp đầu ra
    Dense(10, activation='softmax')  # Lớp đầu ra
])

# Compile mô hình
model_cnn.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Tăng cường dữ liệu (Data Augmentation)

datagen = ImageDataGenerator(
    rotation_range=10,       # Xoay ngẫu nhiên trong khoảng 10 độ
    width_shift_range=0.1,   # Dịch ngang ngẫu nhiên 10%
    height_shift_range=0.1,  # Dịch dọc ngẫu nhiên 10%
    zoom_range=0.1           # Phóng to/thu nhỏ ngẫu nhiên
)

# Áp dụng tăng cường dữ liệu
datagen.fit(x_train)

# Huấn luyện mô hình với tăng cường dữ liệu
history_cnn = model_cnn.fit(datagen.flow(x_train, y_train, batch_size=32),
                            epochs=20,  # Số lượng epoch lớn hơn để tận dụng data augmentation
                            validation_data=(x_test, y_test),
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])  # Early stopping

# Đánh giá mô hình
test_loss_cnn, test_acc_cnn = model_cnn.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc_cnn}")
# Hàm xử lý ảnh
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')                                      # Chuyển ảnh sang grayscale
    img = img.resize((28, 28))                                                     # Resize về 28x28
    img_array = np.array(img)                                                      # Chuyển ảnh thành mảng numpy
    img_array = 255 - img_array                                                    # Đảo màu (nền trắng chữ đen giống MNIST)
    img_array = img_array / 255.0                                                  # Chuẩn hóa giá trị pixel về [0, 1]
    img_array = img_array[np.newaxis, ..., np.newaxis]                             # Thêm chiều batch và channels
    return img_array
# Đường dẫn đến ảnh vẽ tay
image_path = '7.png'
# Xử lý ảnh
input_image = preprocess_image(image_path)
# Dự đoán
prediction = model_cnn.predict(input_image)
predicted_label = np.argmax(prediction, axis=1)[0]
print(f"Dự đoán số viết tay là: {predicted_label}")
import matplotlib.pyplot as plt
# Hiển thị ảnh vẽ tay
plt.imshow(input_image[0, ..., 0], cmap='gray')
plt.title(f"Dự đoán: {predicted_label}")
plt.axis('off')
plt.show()