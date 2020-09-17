#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2 as cv
from std_msgs.msg import String
from sensor_msgs.msg import Image
from PIL import Image as ImageModule
from tesserocr import PyTessBaseAPI, RIL
import numpy as np
import face_recognition
import os
import pickle
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/turtle/camera/rgb/image_raw",Image,self.callback,queue_size=None)
    global known_face_encodings,known_face_names
    known_face_encodings,known_face_names=loadEncodings()

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    cv_image = face_detection(cv_image)

    cv.imshow("Robot View", cv_image)
    cv.waitKey(3)

def text_recognition(im_cv):
  img = cv.cvtColor(im_cv, cv.COLOR_BGR2RGB)
  image = ImageModule.fromarray(img)
  with PyTessBaseAPI() as api:
    api.SetImage(image)
    boxes = api.GetComponentImages(RIL.TEXTLINE, True)
    ocrResult = api.GetUTF8Text()
    #print('Found {} textline image components.'.format(len(boxes)))
    for i, (im, box, _, _) in enumerate(boxes):
      # im is a PIL image object # box is a dict with x, y, w and h keys
      #api.SetRectangle(box['x'], box['y'], box['w'], box['h']) #conf = api.MeanTextConf()
      cv.rectangle(im_cv,(box['x'],box['y']),(box['x']+box['w'],box['y']+box['h']),(0,0,0),1)
    (rows,cols,channels) = im_cv.shape
    height = len(boxes)*25+30
    im_text = np.zeros((height,cols,3), np.uint8)
    im_text[:] = (255,255,255)
    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(im_text,ocrResult,(10,10), font, 1,(0,0,0),1,cv.LINE_AA)
    im_cv = np.concatenate((im_cv, im_text), axis=0)
  return im_cv

def face_detection(frame):
  global known_face_encodings,known_face_names
  face_locations = []
  face_encodings = []
  face_names = []
  small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
  rgb_small_frame = small_frame[:, :, ::-1]
  face_locations = face_recognition.face_locations(rgb_small_frame)
  face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
  for face_encoding in face_encodings:
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    if not face_distances.size == 0:
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
    face_names.append(name)
  for (top, right, bottom, left), name in zip(face_locations, face_names):
    top *= 2
    right *= 2
    bottom *= 2
    left *= 2
    cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    cv.rectangle(frame, (left, bottom), (right, bottom+20), (0, 0, 255), cv.FILLED)
    font = cv.FONT_HERSHEY_DUPLEX
    cv.putText(frame, name, (left + 6, bottom+14), font, 0.5, (255, 255, 255), 1)
  return frame

def loadEncodings():
  filename = '/home/arjun/NTU Project/Face Recognition/data/enc/encodings.pickle'
  with open(filename, 'rb') as f:
    return pickle.load(f)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv.destroyAllWindows()

if __name__ == '__main__':
  known_face_encodings = []
  known_face_names = []
  main(sys.argv)
