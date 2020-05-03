
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

import numpy as np
import cv2
import glob
from utils import label_map_util
from utils import visualization_utils as vis_util
from skimage.io import imsave

MODEL_NAME = 'StampDetection'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('StampDetection', 'label_map.pbtxt')
# Number of classes to detect
NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        image = cv2.imread("./OCR_Test_Images/s1.jpg")
        (height, width) = image.shape[:2]
        image_np_expanded = np.expand_dims(image, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name("detection_boxes:0")
        scores = detection_graph.get_tensor_by_name("detection_scores:0")
        classes = detection_graph.get_tensor_by_name("detection_classes:0")

        num_detections = detection_graph.get_tensor_by_name("num_detections:0")

        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes,num_detections],  feed_dict= {image_tensor: image_np_expanded})
        
        threshold = 0.6

        objects = []
        box = np.squeeze(boxes)

        for index, value in enumerate(classes[0]):
            object_dict = {}
            
            if scores[0, index] > threshold:
                
                ymin = (int(box[index,0]* height))
                xmin = (int(box[index,1]*width))
                ymax = (int(box[index,2]*height))
                xmax = (int(box[index,3]*width))
            
                object_dict[(category_index.get(value)).get('name').encode('utf8')]= ((scores[0, index])*100).astype(int)
                objects.append(object_dict)
                roi = image[ymin:ymax , xmin:xmax].copy()
                cv2.imwrite("./OCR_Output_Images/stamps/OCR_{}.jpg".format(str(index)), roi)
                
                
                #roi = cv2.resize(roi, (250,200))
                #imsave("./OCR_Output_Images/stamps/OCR_{}.jpg".format(str(index)), roi)
                
                  
                
                
        print(objects)
        
        per_file = open("./OCR_Output_Images/stamps/stamps_probability.txt", "w+")
        
        for i in objects:
            for k, v in i.items():
                #print(k,v)
                Tag = str(k)
                Value = str(v)
                final = Tag + " : "+ Value
                per_file.write(final+"\n")
        
        per_file.close()
        
        vis_util.visualize_boxes_and_labels_on_image_array(image, np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)
        
        cv2.imshow("Al Rajhi : KYC Document/", cv2.resize(image, (1000,1000)))
        cv2.waitKey(0)
        for file in glob.glob("./OCR_Output_Images/stamps/*.jpg"):
            cropped_labels = cv2.imread(file)
            cv2.imshow("Detected Labels", cropped_labels)
            cv2.waitKey(0)
        
cv2.destroyAllWindows()
