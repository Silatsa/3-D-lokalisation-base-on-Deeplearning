import numpy as np
import os
import tensorflow as tf
from glob import glob
from PIL import Image
from object_detection.utils import label_map_util
import os
import pyrealsense2 as rs
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import math

pipe = rs.pipeline()
cfg = rs.config()

path_To_BagFile = "../object_detection.bag"

cfg.enable_device_from_file(path_To_BagFile)

profile = pipe.start(cfg)

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
  pipe.wait_for_frames()
  
# Store next frameset for later processing:
frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
depth_frame = frameset.get_depth_frame()

# Cleanup:
pipe.stop()



# Stream Alignment

# Upon closer inspection you can notice that the two frames are not captured from the same physical viewport.

# To combine them into a single RGBD image, let's align depth data to color viewport:

# Create alignment primitive with color as its target stream:  
align = rs.align(rs.stream.color)

frameset2 = align.process(frameset)

#Now the two images are pixel-perfect aligned and you can use depth data just like you would any of the other channels

# Update color and depth frames:
aligned_depth_frame = frameset2.get_depth_frame()

aligned_color_frame = frameset2.get_color_frame()

colorizer = rs.colorizer()
colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
color = np.asanyarray(aligned_color_frame.get_data())
depth = np.asanyarray(depth_frame.get_data())
path_To_Image = '../Distance_To_Object/Image'
cv2.imwrite(os.path.join(path_To_Image , 'waka.png'), color)

cv2.imshow('image', color)
cv2.waitKey(0)
cv2.imshow('image', colorized_depth)
cv2.waitKey(0)

print("Frames Captured")

PATH_TO_CKPT = 'pre-trained-model/frozen_inference_graph.pb'
PATH_TO_LABELS = 'annotations/map.pbtxt'

xmin=0
ymax=0
ymin=0
xmax=0
label =""
NUM_CLASSES = 90

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
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


PATH_TO_TEST_IMAGES_DIR = 'C:/Users/Patrick/Documents/annotation_demo/test'
	  
glob_pattern = os.path.join(PATH_TO_TEST_IMAGES_DIR, '*')
files = sorted(glob(glob_pattern), key=os.path.getctime)
TEST_IMAGE_PATHS = files

	  
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      image_width, image_height = image.size
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      boxes = np.squeeze(boxes)
      classes = np.squeeze(classes)
      scores = np.squeeze(scores)

      #writer = Writer(image_path, image_width, image_height)

      for index, score in enumerate(scores):
        if score < 0.5:
          continue

        label = category_index[classes[index]]['name']
        ymin, xmin, ymax, xmax = boxes[index]

height_color, width_color = color.shape[:2]

height_depth, width_depth = depth.shape[:2]

# if ((height_color >  height_depth or height_color <  height_depth) and (width_color ==  width_depth)):
  # #print("height_color is greater than height_depth")
  
  # scalPixel = abs(int( height_color - height_depth))
  
  # ymin = int(ymin + scalPixel)
  # ymax = int(ymax + scalPixel)
# elif (height_color ==  height_depth) and (width_color <  width_depth or width_color >  width_depth):
  # #print("height_color is greater than height_depth")
  
  # scalPixel = abs(int( width_depth - width_color))
  
  # xmin = int(xmin + scalPixel)
  # xmax = int(xmax + scalPixel)
  
# elif (height_color >  height_depth or height_color <  height_depth) and (width_color <  width_depth or width_color >  width_depth):
  # scalPixel = abs(int( height_color - height_depth))
  
  # ymin = int(ymin + scalPixel)
  # ymax = int(ymax + scalPixel)
  
  # scalPixel = abs(int( width_depth - width_color))
  
  # xmin = int(xmin + scalPixel)
  # xmax = int(xmax + scalPixel)

# boundingbox coordinate
xmin = int(xmin * width_color)
ymin = int(ymin * height_color)
xmax = int(xmax * width_color)
ymax =  int(ymax * height_color)


colorized_depth2 = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data()) 

#print boundingbox coordinate 
print("xmin: %d\n"% int(xmin))
print("ymin: %d\n"% int(ymin))
print("xmax: %d\n"% int(xmax))
print("ymax: %d\n"% int(ymax))


cv2.rectangle(colorized_depth2, (xmin, ymin), 
             (xmax, ymax), (255, 255, 255), 2)
cv2.imshow('image', colorized_depth2)
cv2.waitKey(0)			 
			 
# cv2.rectangle(color, (15, 0), 
             # (504, 720), (255, 255, 255), 2)

# cv2.imshow('image', color)
# cv2.waitKey(0)

# Downcast the frames to video_stream_profile and fetch intrinsics
depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics


Hypotenuse_max = aligned_depth_frame.get_distance(xmax,ymin)

Hypotenuse_min = aligned_depth_frame.get_distance(xmin,ymax)


depth_pixel_y_min = [xmax,ymin]

depth_value_y_min = aligned_depth_frame.get_distance(int(xmin), int(ymin))

#maps the pixel [xmin,ymin] to a 3D point location within the stream's associated 3D coordinate space
depth_point_y_min= rs.rs2_deproject_pixel_to_point(depth_intrin,depth_pixel_y_min,depth_value_y_min)

depth_pixel_y_max = [xmax,ymax]
depth_value_y_max = aligned_depth_frame.get_distance(int(xmax), int(ymax))

#maps the pixel [xmax,ymax] to a 3D point location within the stream's associated 3D coordinate space
depth_point_y_max = rs.rs2_deproject_pixel_to_point(depth_intrin,depth_pixel_y_max,depth_value_y_max)

y = (depth_point_y_max[1] - depth_point_y_min[1])

x = (depth_point_y_max[0] - depth_point_y_min[0])

# horizontal distance from camera to detected object
dist = math.sqrt( abs( - (math.pow(y, 4)- 2 * math.pow(y, 2) * math.pow(Hypotenuse_min, 2) - 2*math.pow(y, 2)*math.pow(Hypotenuse_max, 2) + math.pow(Hypotenuse_max, 4)
 + math.pow(Hypotenuse_min, 4)- 2 * math.pow(Hypotenuse_max, 2) * math.pow(Hypotenuse_min, 2)) / (4 * math.pow(y, 2))))


print("y  %8.2f\n"% (depth_point_y_max[1] - depth_point_y_min[1]))

print("Hypotenuse_min %8.2f\n"% Hypotenuse_min)
print("Hypotenuse_max %8.2f\n"% Hypotenuse_max)

print("Detected %8.2f meters away.\n"% dist)

# angle horizontal distance and hypotenuse in degrees
angle_for_ymin = math.degrees(math.acos( dist / Hypotenuse_min ))

angle_for_ymax = math.degrees(math.acos( dist / Hypotenuse_max))

		

