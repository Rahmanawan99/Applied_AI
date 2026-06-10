import cv2 
import torch 
import joblib 
import numpy as np 
import time 
from PIL import Image 
from facenet_pytorch import InceptionResnetV1 ,MTCNN 
from liveness import BlinkDetector 
from attendance import mark_attendance 
import collections 




device ='cuda'if torch .cuda .is_available ()else 'cpu'
print (f"Using device: {device }")

mtcnn =MTCNN (image_size =160 ,margin =20 ,device =device )
resnet =InceptionResnetV1 (pretrained ='vggface2').eval ().to (device )

try :
    model =joblib .load ("models/face_classifier.pkl")
    print ("Model loaded successfully.")
except Exception as e :
    print (f"Error loading model: {e }")
    exit ()

blink_detector =BlinkDetector ()
THRESHOLD =0.7 


ear_history =collections .deque (maxlen =100 )
confidence_history =collections .deque (maxlen =100 )

def draw_hud (frame ,name ,confidence ,ear ,attendance_marked ):
    h ,w ,_ =frame .shape 


    overlay =frame .copy ()
    cv2 .rectangle (overlay ,(0 ,0 ),(w ,120 ),(0 ,0 ,0 ),-1 )
    cv2 .addWeighted (overlay ,0.6 ,frame ,0.4 ,0 ,frame )


    status_color =(0 ,255 ,0 )if confidence >THRESHOLD else (0 ,0 ,255 )


    cv2 .putText (frame ,f"USER: {name .upper ()}",(20 ,40 ),cv2 .FONT_HERSHEY_DUPLEX ,0.8 ,(255 ,255 ,255 ),1 )
    cv2 .putText (frame ,f"CONFIDENCE:",(20 ,75 ),cv2 .FONT_HERSHEY_DUPLEX ,0.6 ,(200 ,200 ,200 ),1 )


    bar_width =int (200 *confidence )
    cv2 .rectangle (frame ,(150 ,60 ),(350 ,80 ),(50 ,50 ,50 ),-1 )
    cv2 .rectangle (frame ,(150 ,60 ),(150 +bar_width ,80 ),status_color ,-1 )
    cv2 .putText (frame ,f"{confidence :.2f}",(360 ,75 ),cv2 .FONT_HERSHEY_DUPLEX ,0.6 ,status_color ,1 )


    cv2 .putText (frame ,f"LIVENESS (EAR):",(20 ,105 ),cv2 .FONT_HERSHEY_DUPLEX ,0.6 ,(200 ,200 ,200 ),1 )
    ear_bar_width =int (200 *(ear *2 ))
    cv2 .rectangle (frame ,(150 ,90 ),(350 ,110 ),(50 ,50 ,50 ),-1 )
    cv2 .rectangle (frame ,(150 ,90 ),(150 +min (ear_bar_width ,200 ),110 ),(255 ,255 ,0 ),-1 )


    if attendance_marked :
        cv2 .rectangle (frame ,(w //2 -150 ,h //2 -30 ),(w //2 +150 ,h //2 +30 ),(0 ,255 ,255 ),-1 )
        cv2 .putText (frame ,"ATTENDANCE MARKED",(w //2 -130 ,h //2 +10 ),cv2 .FONT_HERSHEY_DUPLEX ,0.8 ,(0 ,0 ,0 ),2 )

def draw_charts (ear_hist ,conf_hist ):

    chart_w ,chart_h =400 ,400 
    chart_img =np .zeros ((chart_h ,chart_w ,3 ),dtype =np .uint8 )


    cv2 .putText (chart_img ,"STABILITY & QUALITY METRICS",(20 ,30 ),cv2 .FONT_HERSHEY_SIMPLEX ,0.6 ,(255 ,255 ,255 ),1 )


    cv2 .putText (chart_img ,"Blink EAR Stability",(20 ,70 ),cv2 .FONT_HERSHEY_SIMPLEX ,0.5 ,(255 ,255 ,0 ),1 )
    if len (ear_hist )>2 :
        for i in range (len (ear_hist )-1 ):
            y1 =int (180 -ear_hist [i ]*200 )
            y2 =int (180 -ear_hist [i +1 ]*200 )
            cv2 .line (chart_img ,(20 +i *3 ,y1 ),(20 +(i +1 )*3 ,y2 ),(255 ,255 ,0 ),1 )


    cv2 .putText (chart_img ,"Face Quality (Confidence)",(20 ,230 ),cv2 .FONT_HERSHEY_SIMPLEX ,0.5 ,(0 ,255 ,0 ),1 )
    if len (conf_hist )>2 :
        for i in range (len (conf_hist )-1 ):
            y1 =int (340 -conf_hist [i ]*100 )
            y2 =int (340 -conf_hist [i +1 ]*100 )
            cv2 .line (chart_img ,(20 +i *3 ,y1 ),(20 +(i +1 )*3 ,y2 ),(0 ,255 ,0 ),1 )

    return chart_img 




cap =cv2 .VideoCapture (0 )
attendance_alert_timer =0 

print ("Starting Advanced GUI... Press 'q' to quit.")

while True :
    success ,frame =cap .read ()
    if not success :break 


    rgb =cv2 .cvtColor (frame ,cv2 .COLOR_BGR2RGB )
    img =Image .fromarray (rgb )
    face =mtcnn (img )

    current_name ="Detecting..."
    current_conf =0.0 
    blink_triggered =False 


    blink_triggered ,current_ear =blink_detector .detect_blink (frame )
    ear_history .append (current_ear )

    if face is not None :
        face =face .unsqueeze (0 ).to (device )
        embedding =resnet (face ).detach ().cpu ().numpy ()
        probs =model .predict_proba (embedding )[0 ]
        current_conf =np .max (probs )
        classes =model .named_steps ['kneighborsclassifier'].classes_ 
        current_name =classes [np .argmax (probs )]

        if current_conf >0.6 and blink_triggered :
            mark_attendance (current_name )
            attendance_alert_timer =30 

    confidence_history .append (current_conf )


    draw_hud (frame ,current_name ,current_conf ,current_ear ,attendance_alert_timer >0 )
    if attendance_alert_timer >0 :attendance_alert_timer -=1 

    charts =draw_charts (ear_history ,confidence_history )


    combined_img =np .hstack ((frame ,cv2 .resize (charts ,(frame .shape [0 ],frame .shape [0 ]))))

    cv2 .imshow ('Applied AI - Pro Attendance System',combined_img )

    if cv2 .waitKey (1 )&0xFF ==ord ('q'):break 

cap .release ()
cv2 .destroyAllWindows ()
