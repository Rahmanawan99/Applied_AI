import os 
import cv2 
import torch 
import numpy as np 

from PIL import Image 
from facenet_pytorch import InceptionResnetV1 ,MTCNN 

from sklearn .svm import SVC 
from sklearn .model_selection import train_test_split 
from sklearn .metrics import (
accuracy_score ,
precision_score ,
recall_score ,
f1_score ,
confusion_matrix ,
classification_report ,
roc_curve ,
auc ,
precision_recall_curve 
)
import joblib 
import matplotlib .pyplot as plt 
import numpy as np 

device ='cuda'if torch .cuda .is_available ()else 'cpu'

print ("Using device:",device )


mtcnn =MTCNN (image_size =160 ,margin =20 ,device =device )


resnet =InceptionResnetV1 (pretrained ='vggface2').eval ().to (device )

dataset_path ="dataset"

X =[]
y =[]

for person_name in os .listdir (dataset_path ):
    if person_name .lower ()=='test':
        continue 
    person_folder =os .path .join (dataset_path ,person_name )

    for image_name in os .listdir (person_folder ):
        if not image_name .lower ().endswith (('.png','.jpg','.jpeg','.bmp','.gif')):
            continue 
        image_path =os .path .join (person_folder ,image_name )
        img =Image .open (image_path ).convert ('RGB')
        face =mtcnn (img )

        if face is not None :
            face =face .unsqueeze (0 ).to (device )
            embedding =resnet (face )
            X .append (embedding .detach ().cpu ().numpy ()[0 ])
            y .append (person_name )

X =np .array (X )
y =np .array (y )


unique_classes ,counts =np .unique (y ,return_counts =True )
valid_classes =unique_classes [counts >=2 ]
mask =np .isin (y ,valid_classes )

if len (unique_classes )!=len (valid_classes ):
    skipped =set (unique_classes )-set (valid_classes )
    print (f"WARNING: Skipping classes with < 2 images: {skipped }")

X =X [mask ]
y =y [mask ]

print ("Final Embeddings shape:",X .shape )

from sklearn .preprocessing import StandardScaler 
from sklearn .pipeline import make_pipeline 


X_train ,X_test ,y_train ,y_test =train_test_split (
X ,
y ,
test_size =0.2 ,
random_state =42 ,
stratify =y 
)

from sklearn .neighbors import KNeighborsClassifier 


model =make_pipeline (
StandardScaler (),
KNeighborsClassifier (n_neighbors =3 ,weights ='distance',metric ='cosine')
)
model .fit (X_train ,y_train )


predictions =model .predict (X_test )
accuracy =accuracy_score (y_test ,predictions )
print ("Accuracy:",accuracy )


joblib .dump (model ,"models/face_classifier.pkl")
print ("Model saved!")


y_prob =model .predict_proba (X_test )
threshold =0.6 
y_pred =[]
classes =model .named_steps ['kneighborsclassifier'].classes_ 

for prob in y_prob :
    confidence =max (prob )
    if confidence >threshold :
        prediction =classes [np .argmax (prob )]
    else :
        prediction ="Unknown"
    y_pred .append (prediction )
    print (f"Prediction: {prediction } | Confidence: {confidence :.2f}")



print (classification_report (y_test ,y_pred ))

print ("\nAccuracy:",accuracy_score (y_test ,y_pred ))
print ("Precision:",precision_score (y_test ,y_pred ,average ='weighted'))
print ("Recall:",recall_score (y_test ,y_pred ,average ='weighted'))
print ("F1 Score:",f1_score (y_test ,y_pred ,average ='weighted'))

cm =confusion_matrix (y_test ,y_pred )

plt .figure ()
plt .imshow (cm )
plt .title ("Confusion Matrix")
plt .xlabel ("Predicted")
plt .ylabel ("Actual")
plt .colorbar ()
plt .show ()

from sklearn .preprocessing import label_binarize 

y_test_bin =label_binarize (y_test ,classes =np .unique (y_test ))

fpr ,tpr ,_ =roc_curve (y_test_bin [:,0 ],y_prob [:,0 ])
roc_auc =auc (fpr ,tpr )

plt .figure ()
plt .plot (fpr ,tpr ,label =f"AUC = {roc_auc :.2f}")
plt .plot ([0 ,1 ],[0 ,1 ],linestyle ="--")
plt .xlabel ("False Positive Rate")
plt .ylabel ("True Positive Rate")
plt .title ("ROC Curve")
plt .legend ()
plt .show ()

precision ,recall ,_ =precision_recall_curve (y_test_bin [:,0 ],y_prob [:,0 ])

plt .figure ()
plt .plot (recall ,precision )
plt .xlabel ("Recall")
plt .ylabel ("Precision")
plt .title ("Precision-Recall Curve")
plt .show ()