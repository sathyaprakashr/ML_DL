from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer
from tensorflow import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    RocCurveDisplay
import matplotlib.pyplot as plt
import os

in_dir = "C:/project/data/english_data/data_cv/"
datagen = ImageDataGenerator(validation_split=0.3)
train_generator = datagen.flow_from_directory(in_dir, target_size=(28, 28), subset='training')
val_generator = datagen.flow_from_directory(in_dir, target_size=(28, 28), subset='validation', shuffle=False)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 3)))
model.add(MaxPool2D(strides=2))
model.add(Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'))
model.add(MaxPool2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(26, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=5e-4))
model.build()
# model.summary()
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.2, min_lr=1e-6)

for ep in [10,15,20,25,30]:
    der = "C:/project/New folder/LeNet/epoch=%d" % (ep)
    history = model.fit(train_generator, epochs=ep, validation_data=val_generator, callbacks=[reduce_lr])

    acc = model.evaluate(val_generator)[1]
    print('model accuracy: {}'.format(round(acc, 4)))

    dir = "C:/project/data/english_data/test_img/"
    datagen = ImageDataGenerator()
    test_generator = datagen.flow_from_directory(dir, target_size=(28, 28), shuffle=False)

    pred = model.predict(test_generator)
    print(pred.shape)
    p = np.argmax(pred, axis=1)

    #--------------------------------------------------------------------------------

    def mak_dir(out):  # for creating a dir
        path = os.path.join("C:/project/New folder/LeNet/epoch=%i" % (out))
        if not os.path.exists(path):
            os.mkdir(path)


    mak_dir(ep)
    #--------------------------------------------------------------------------------

    print("\n\nepoch=%d" % (ep))
    print(model.summary())
    acc = model.evaluate(val_generator)[1]
    print('model accuracy%:', acc * 100)

    ac = accuracy_score(p, test_generator.labels)
    print("accuracy%:", ac * 100)

    pr = precision_score(p, test_generator.labels, average='micro')
    print("precision_score_micro%:", pr * 100)
    pr = precision_score(p, test_generator.labels, average='weighted')
    print("precision_score_weighted%:", pr * 100)
    pr = precision_score(p, test_generator.labels, average='macro')
    print("precision_score_macro%:", pr * 100)

    r = recall_score(p, test_generator.labels, average='micro')
    print("recall_score% :", r * 100)

    f = f1_score(test_generator.labels, p, average='micro')
    print("f1_score%:", f * 100)
    # --------------------------------------------------------------------------------
    plt.close('all')
    cmd = ConfusionMatrixDisplay.from_predictions(p, test_generator.labels)
    plt.title("Confusion Matrix")
    plt.show()
    plt.close()

    # --------------------------------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.title("Loss Graph")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig(der + "/Loss_Graph.png")

    # --------------------------------------------------------------------------------
    # plot the accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.title("Accuracy Graph")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(der + "/Accuracy_Graph.png")

    # --------------------------------------------------------------------------------
    alpha = "abcdefghijklmnopqrstuvwxyz"
    train_labels = train_generator.classes
    train_labels = [alpha[i] for i in train_labels]

    val_labels = test_generator.classes
    val_labels = [alpha[i] for i in val_labels]

    label_binarizer = LabelBinarizer().fit(train_labels)
    y_onehot_test = label_binarizer.transform(val_labels)
    var = y_onehot_test.shape  # (n_samples, n_classes)
    print(var)
    ig, ax = plt.subplots(figsize=(12, 8))

    cmap = plt.get_cmap("tab20")
    num_colors = 20

    for i, class_of_interest in enumerate(alpha):
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]
        display = RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            pred[:, class_id],
            name=f"{class_of_interest} vs the rest",
            color=cmap(i % num_colors),
            ax=ax
        )

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC_curve")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(der + "/ROC_curve.png")

