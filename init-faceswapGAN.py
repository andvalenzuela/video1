global TOTAL_ITERS
TOTAL_ITERS = 25000

import imageio
imageio.plugins.ffmpeg.download()
import keras.backend as K
from detector.face_detector import MTCNNFaceDetector
import glob
from preprocess import preprocess_video

%tensorflow_version 1.x
import tensorflow as tf
print(tf.__version__)

fd = MTCNNFaceDetector(sess=K.get_session(), model_path="./mtcnn_weights/")

save_interval = 5 # perform face detection every {save_interval} frames
save_path = "./faceA/"
preprocess_video(fn_source_video, fd, save_interval, save_path)
save_path = "./faceB/"
preprocess_video(fn_target_video, fd, save_interval, save_path)

print(str(len(glob.glob("faceA/rgb/*.*"))) + " face(s) extracted from source video: " + fn_source_video + ".")
print(str(len(glob.glob("faceB/rgb/*.*"))) + " face(s) extracted from target video: " + fn_target_video + ".")
