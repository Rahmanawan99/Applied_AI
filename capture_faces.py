import cv2 
import os 

save_path ="dataset/rahman"

os .makedirs (save_path ,exist_ok =True )

cap =cv2 .VideoCapture (0 )

count =0 

while True :
    ret ,frame =cap .read ()
    cv2 .imshow ("Capture Faces",frame )
    key =cv2 .waitKey (1 )

    if key ==ord ('s'):
        img_path =f"{save_path }/{count }.jpg"
        cv2 .imwrite (img_path ,frame )
        print (f"Saved {img_path }")
        count +=1 

    elif key ==ord ('q'):
        break 

cap .release ()
cv2 .destroyAllWindows ()