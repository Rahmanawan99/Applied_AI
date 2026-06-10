import numpy as np 
import torch 
import mediapipe as mp 

print ("NumPy:",np .__version__ )
print ("CUDA:",torch .cuda .is_available ())
print ("MediaPipe:",hasattr (mp ,"solutions"))