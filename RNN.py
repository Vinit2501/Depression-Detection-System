import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Flatten, Dropout, LSTM, Bidirectional, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
from tensorflow.keras.models import save_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


# Function to create DataFrame from folder containing audio files
def create_dataframe(main_folder):
    data = []
    for folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, folder)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            stress_level = folder
            data.append({'file_path': file_path, 'stress_level': stress_level})
    df = pd.DataFrame(data)
    return df


def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.reshape(-1, 1))
    return scaled_features.flatten()

def convert_stress_levels(df):
    df = pd.get_dummies(df)
    return df

def add_noise(y, noise_factor=0.17):
    noise = np.random.normal(0, noise_factor, size=y.shape)
    noisy_y = y + noise
    return noisy_y


# Function to extract MFCC features from audio file
def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    y_noisy = add_noise(y)
    mfcc = librosa.feature.mfcc(y=y_noisy, sr=sr)
    mfcc_mean = np.mean(mfcc, axis=1)
    return scale_features(mfcc_mean)

# Function to extract features for all audio files in the DataFrame
def extract_features_from_dataframe(df):
    X = []
    y = []
    for index, row in df.iterrows():
        audio_file = row['file_path']

        stress_level = row['stress_level']
        features = extract_features(audio_file)
        X.append(features)
        y.append(stress_level)
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    return X, y


def load_dataframe():
    root_dir = "C:\\Users\\win10\\OneDrive\\Desktop\\Final year\\Dataset"
    df = create_dataframe(root_dir)
    df = shuffle(df, random_state=42)
    print(df)
    return df

# Load DataFrame
df = load_dataframe()

# Extract features from audio files in DataFrame
X, y = extract_features_from_dataframe(df)
y = convert_stress_levels(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model architecture
model = Sequential()
model.add(Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(128, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=5, strides=2, padding='same'))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
model.add(Dropout(0.2))

model.add(Bidirectional(LSTM(units=64)))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax'))


# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)
y_test_class = np.argmax(y_test, axis=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy:.3f}')
print(f'Test loss: {loss:.3f}')

print("Accuracy of our model on test data : " , model.evaluate(X_test,y_test)[1]*100 , "%")
print()
print(classification_report(y_test_class, y_pred_class))

#Visualize the loss and accuracy graphs
epochs = [i for i in range(50)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
test_acc = history.history['val_accuracy']
test_loss = history.history['val_loss']

fig.set_size_inches(12,6)
ax[0].plot(epochs , train_loss , label = 'Training Loss')
ax[0].plot(epochs , test_loss , label = 'Testing Loss')
ax[0].set_title('Training & Testing Loss')
ax[0].legend()
ax[0].set_xlabel("Epochs")

ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
ax[1].set_title('Training & Testing Accuracy')
ax[1].legend()
ax[1].set_xlabel("Epochs")
plt.show()

y_pred = model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

y_test_class = np.argmax(y_test, axis=1)
conf_mat = confusion_matrix(y_test_class, y_pred_class)

# Visualize the confusion matrix
plt.imshow(conf_mat, interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.show()

# Save the model as an .h5 file
save_model(model, 'stress_detection_model.h5')
 