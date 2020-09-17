#!/usr/bin/python

import Tkinter
import roslaunch
import subprocess
import rosbag
import tkFileDialog
import os
import cv2
from Tkinter import *
import signal

global fname


top = Tkinter.Tk()
# Code to add widgets will go here...
top.title('Visualization')
top.geometry('300x300')
top.configure(background='#0000FF')


def launched0():
  os.kill(os.getppid(), signal.SIGHUP)



def launched1():
  uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
  roslaunch.configure_logging(uuid)
  launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/tanmay/catkin_ws/src/rtabmap_ros/launch/demo/demo_stereo_outdoor.launch"])
  launch.start()

  launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/tanmay/catkin_ws/src/rtabmap_ros/launch/bgpl.launch"])
  launch.start()
  top.after(2000,lambda:top2)
  

  

def launched2():
  top2 = Tkinter.Tk()
# Code to add widgets will go here...
  top2.title('Bag File')
  top2.geometry('300x300')
  top2.configure(background='#0000FF')
  
  def launched3():
	  cv2.namedWindow("preview")
	  vc = cv2.VideoCapture(0)

	  if vc.isOpened(): # try to get the first frame
		rval, frame = vc.read()
	  else:
		rval = False

	  while rval:
	 	cv2.imshow("preview", frame)
	 	rval, frame = vc.read()
	 	key = cv2.waitKey(20)
	    	if key == 27: # exit on ESC
			break
	  cv2.destroyWindow("preview")
	  top2.destroy()
	  
   
  def launched4():
	fname = tkFileDialog.askopenfilename(filetypes = (("Bag Files", "*.bag"), ("All files", "*")))
	top2.destroy()

  e = Tkinter.Button(top2,text ="use camera", command = launched3, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
  f = Tkinter.Button(top2,text ="Upload from device", command = launched4, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
  e.pack()
  e.place(x=60,y=50)
  f.pack()
  f.place(x=60,y=150)

  
  
  

B = Tkinter.Button(top,text ="Launch Simulation", command = launched1, height=1, width=20,fg= '#000000',bg ='#FFFFFF')

D = Tkinter.Button(top,text ="Select Bag File", command = launched2, height=1, width=20,fg= '#000000',bg ='#FFFFFF')

G = Tkinter.Button(top,text ="Stop", command = launched0, height=1, width=20,fg= '#000000',bg ='#FFFFFF')

B.pack()
B.place(x=60,y=50)
D.pack()
D.place(x=60,y=150)
G.pack()
G.place(x=60,y=250)
top.mainloop()
