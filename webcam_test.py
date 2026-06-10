import cv2 
import torch 
import joblib 
import numpy as np 

from PIL import Image 
from facenet_pytorch import InceptionResnetV1 ,MTCNN 

device ='cuda'if torch .cuda .is_available ()else 'cpu'

mtcnn =MTCNN (image_size =160 ,margin =20 ,device =device )

resnet =InceptionResnetV1 (pretrained ='vggface2').eval ().to (device )

model =joblib .load ("models/face_classifier.pkl")

cap =cv2 .VideoCapture (0 )

while True :
    ret ,frame =cap .read ()
    rgb =cv2 .cvtColor (frame ,cv2 .COLOR_BGR2RGB )
    img =Image .fromarray (rgb )
    face =mtcnn (img )

    if face is not None :
        face =face .unsqueeze (0 ).to (device )
        embedding =resnet (face )
        embedding =embedding .detach ().cpu ().numpy ()
        prediction =model .predict (embedding )

        cv2 .putText (
        frame ,
        prediction [0 ],
        (20 ,50 ),
        cv2 .FONT_HERSHEY_SIMPLEX ,
        1 ,
        (0 ,255 ,0 ),
        2 
        )

    cv2 .imshow ("FaceID Attendance",frame )

    if cv2 .waitKey (1 )&0xFF ==ord ('q'):
        break 

cap .release ()
cv2 .destroyAllWindows ()