import cv2
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from tensorflow import *
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, \
    RocCurveDisplay
import matplotlib.pyplot as plt
import os

in_dir="C:/project/data/english_data/data_cv/"
l="abcdefhghijklmnopqrstuvwxyz"
train,val,lt,lv=[],[],[],[]
der = "C:/project/New folder/Xgboost/"

for i in l:
  l1,l2=[],[]
  a=in_dir+i
  size=len(os.listdir(a))
  s1=int(size*0.7)
  s2=size-s1
  for j in os.listdir(a):
    #print(os.path.exists(os.path.join(a,j)))
    h=cv2.imread(os.path.join(a,j),0)
    if s1>0:
      train.append(h)
      lt.append(i)
      s1-=1
    else :
      val.append(h)
      lv.append(i)

train_gn=np.array([np.array(i) for i in train])
val_gn=np.array([np.array(i) for i in val])

lt_g=np.array(lt)
lv_g=np.array(lv)

print(val_gn.shape)
print(train_gn.shape)
print(lv_g.shape)

le=LabelEncoder()
lt=le.fit_transform(lt_g)
lv=le.fit_transform(lv_g)

train=train_gn.reshape(2643,28*28)
val=val_gn.reshape(1134,28*28)

#m=xgb.XGBClassifier(max_depth=10,booster='gblinear',min_child_weight=5,max_delta_step=5)
#minchild weight=<5,booster-glinear,max_depth=7-10 or maxleafnodes=0-5,maxdelta step=<5

m=xgb.XGBClassifier(base_score=0.2, booster='gblinear', learning_rate=0.1,n_estimators=200,objective='multi:softprob',verbosity=1)

m.fit(train,lt)

p=m.predict(val)
print(p)

acc = accuracy_score(lv,p)
print('model accuracy%:', acc * 100)

pr = precision_score(p, lv, average='micro')
print("precision_score_micro%:", pr * 100)
pr = precision_score(p, lv, average='weighted')
print("precision_score_weighted%:", pr * 100)
pr = precision_score(p, lv, average='macro')
print("precision_score_macro%:", pr * 100)

r = recall_score(p, lv, average='micro')
print("recall_score% :", r * 100)

f = f1_score(lv, p, average='micro')
print("f1_score%:", f * 100)
# --------------------------------------------------------------------------------
plt.close('all')
cmd = ConfusionMatrixDisplay.from_predictions(p, lv)
plt.title("Confusion Matrix")
plt.show()
plt.close()

# --------------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
plt.plot(m.history['loss'], label='train loss')
plt.plot(m.history['val_loss'], label='val loss')
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig(der + "/Loss_Graph.png")

# --------------------------------------------------------------------------------
# plot the accuracy
plt.figure(figsize=(8, 6))
plt.plot(m.history['accuracy'], label='train acc')
plt.plot(m.history['val_accuracy'], label='val acc')
plt.legend()
plt.title("Accuracy Graph")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.savefig(der + "/Accuracy_Graph.png")

# --------------------------------------------------------------------------------
alpha = "abcdefghijklmnopqrstuvwxyz"
train_labels = lt
train_labels = [alpha[i] for i in train_labels]

val_labels = lv
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
    p[:, class_id],
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