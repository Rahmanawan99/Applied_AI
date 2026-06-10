import os 
import torch 
from PIL import Image 
from facenet_pytorch import MTCNN 


device ='cuda'if torch .cuda .is_available ()else 'cpu'
mtcnn =MTCNN (keep_all =False ,device =device )

test_dir ='test'
output_dir ='test_cropped'

if not os .path .exists (output_dir ):
    os .makedirs (output_dir )

print (f"Starting face cropping in '{test_dir }' using {device }...")

count =0 
for filename in os .listdir (test_dir ):
    if filename .lower ().endswith (('.png','.jpg','.jpeg')):
        img_path =os .path .join (test_dir ,filename )
        try :
            img =Image .open (img_path ).convert ('RGB')


            boxes ,_ =mtcnn .detect (img )

            if boxes is not None :

                box =boxes [0 ]
                x1 ,y1 ,x2 ,y2 =[int (b )for b in box ]


                w ,h =x2 -x1 ,y2 -y1 
                margin_w ,margin_h =int (w *0.1 ),int (h *0.1 )

                x1 =max (0 ,x1 -margin_w )
                y1 =max (0 ,y1 -margin_h )
                x2 =min (img .width ,x2 +margin_w )
                y2 =min (img .height ,y2 +margin_h )


                face_img =img .crop ((x1 ,y1 ,x2 ,y2 ))
                face_img .save (os .path .join (output_dir ,filename ))
                count +=1 
                if count %10 ==0 :
                    print (f"Processed {count } images...")
            else :
                print (f"Skipping {filename }: No face detected.")
        except Exception as e :
            print (f"Error processing {filename }: {e }")

print (f"\n✅ Done! Successfully cropped {count } faces into '{output_dir }'.")
