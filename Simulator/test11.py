import Tkinter
import subprocess
import tkFileDialog
import os
from Tkinter import *
import signal
import cv2
import roslaunch
import rosbag

global fname


main = Tkinter.Tk()
main.title('OS Mind')
main.geometry('300x350')
main.configure(background='#0000FF')

def virtual_cmd():
	main2 = Tkinter.Tk()
	main2.title('Virtual World')
	main2.geometry('300x350')
	main2.configure(background='#0000FF')
	main.withdraw()

	def d2d_cmd():
		main10 = Tkinter.Tk()
		main10.title('2D World')
		main10.geometry('300x350')
		main10.configure(background='#0000FF')
		#Arjun code 2d gazebo

	def d3d_cmd():
		main3 = Tkinter.Tk()
		main3.title('3D World')
		main3.geometry('300x350')
		main3.configure(background='#0000FF')
		main2.withdraw()

		def bag_cmd():
			main4 = Tkinter.Tk()
			main4.title('Select Bag File')
			main4.geometry('300x350')
			main4.configure(background='#0000FF')
			main3.withdraw()

			def new_bag_cmd():
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

			def existing_bag_cmd():
			        fname = tkFileDialog.askopenfilename(filetypes = (("Bag Files", "*.bag"), ("All files", "*")))
				main5 = Tkinter.Tk()
				main5.title('Choose Option')
				main5.geometry('300x500')
				main5.configure(background='#0000FF')
				main4.withdraw()

				def launch_vis_cmd():
					uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
					roslaunch.configure_logging(uuid)
					launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/tanmay/catkin_ws/src/rtabmap_ros/launch/demo/demo_stereo_outdoor.launch"])
					launch.start()
                                        launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/tanmay/catkin_ws/src/rtabmap_ros/launch/bgpl.launch"])
					launch.start()
					
					main6 = Tkinter.Tk()
					main6.title('Visualisation')
					main6.geometry('300x500')
					main6.configure(background='#0000FF')
					
					def record_cmd():
						uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
						roslaunch.configure_logging(uuid)
						launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/tanmay/catkin_ws/src/rtabmap_ros/launch/record.launch"])
						launch.start()
						
					def stop_rec_cmd():
						os.system("rosnode kill /recorded")
						
					def stop_vis_cmd():
						os.system("rosnode kill /rviz")
  					        os.system("rosnode kill /player")	
					
					record = Tkinter.Button(main6,text ="Record", command = record_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
					record.pack()
					record.place(x=60,y=150)
					stop_rec = Tkinter.Button(main6,text ="Stop Recording", command = stop_rec_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
					stop_rec.pack()
					stop_rec.place(x=60,y=250)
					stop_vis = Tkinter.Button(main6,text ="Stop Visualisation", command = stop_vis_cmd , height=1, width=20,fg= '#000000',bg ='#FFFFFF')
					stop_vis.pack()
					stop_vis.place(x=60,y=350)
				
				def launch_sim_cmd():
					os.system("cd ~/Downloads")
				 	os.system("wget https://bitbucket.org/osrf/gazebo_tutorials/raw/default/dem/files/mtsthelens_before.zip")
				 	os.system("unzip ~/Downloads/mtsthelens_before.zip -d /tmp")
				 	os.system("mv /tmp/30.1.1.1282760.dem /tmp/mtsthelens.dem")
				 	os.system("mkdir -p /tmp/media/dem/")
				 	os.system("gdalwarp -ts 129 129 /tmp/mtsthelens.dem /tmp/media/dem/mtsthelens_129.dem")
				 	#os.execl(["/usr/share/gazebo", "source setup.sh"])
				 	os.system("GAZEBO_RESOURCE_PATH='$GAZEBO_RESOURCE_PATH:/tmp'")
				 	os.system("gazebo /tmp/volcano.world")
				 	
				def launch_gmap_cmd():
					main7 = Tkinter.Tk()
					main7.title('Gmapping')
					main7.geometry('300x500')
					main7.configure(background='#0000FF')
					
				launch_vis = Tkinter.Button(main5,text ="Launch Visualisation", command = launch_vis_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
				launch_vis.pack()
				launch_vis.place(x=60,y=125)
				launch_sim = Tkinter.Button(main5,text ="Launch Simulation", command = launch_sim_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
				launch_sim.pack()
				launch_sim.place(x=60,y=225)
				launch_gmap = Tkinter.Button(main5,text ="Launch Gmapping", command = launch_gmap_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
				launch_gmap.pack()
				launch_gmap.place(x=60,y=325)
				
			new_bag = Tkinter.Button(main4,text ="Create new Bag File", command = new_bag_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
			new_bag.pack()
			new_bag.place(x=60,y=100)
			exisiting_bag = Tkinter.Button(main4,text ="Select from Existing Bag File", command = existing_bag_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
			exisiting_bag.pack()
			exisiting_bag.place(x=60,y=200)


		bag = Tkinter.Button(main3,text ="Select Bag File", command = bag_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
		bag.pack()
		bag.place(x=60,y=150)
		

	d2d = Tkinter.Button(main2,text ="2D", command = d2d_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
	d3d = Tkinter.Button(main2,text ="3D", command = d3d_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
	d2d.pack()
	d2d.place(x=60,y=100)
	d3d.pack()
	d3d.place(x=60,y=200)



def real_cmd():
	main11 = Tkinter.Tk()
	main11.title('Real World')
	main11.geometry('300x350')
	main11.configure(background='#0000FF')
	main.withdraw()	

'''top = Tkinter.Tk()
# Code to add widgets wsill go here...
top.title('OS Mind')
top.geometry('300x500')
top.configure(background='#0000FF')


def launched0():
  os.system("rosnode kill /rviz")
  os.system("rosnode kill /player")



def launch_sim():
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

def launched5():
	uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
	roslaunch.configure_logging(uuid)
	launch = roslaunch.parent.ROSLaunchParent(uuid, ["/home/tanmay/catkin_ws/src/rtabmap_ros/launch/record.launch"])
	launch.start()
	

def launched6():
  os.system("rosnode kill /recorded")

def launched7(): 
#  os.system("cd ~/Downloads")
#  os.system("wget https://bitbucket.org/osrf/gazebo_tutorials/raw/default/dem/files/mtsthelens_before.zip")
#  os.system("unzip ~/Downloads/mtsthelens_before.zip -d /tmp")
#  os.system("mv /tmp/30.1.1.1282760.dem /tmp/mtsthelens.dem")
#  os.system("mkdir -p /tmp/media/dem/")
#  os.system("gdalwarp -ts 129 129 /tmp/mtsthelens.dem /tmp/media/dem/mtsthelens_129.dem")
 # os.execl(["/usr/share/gazebo", "source setup.sh"])
  os.system("GAZEBO_RESOURCE_PATH='$GAZEBO_RESOURCE_PATH:/tmp'")
  os.system("gazebo /tmp/volcano.world")



virtual  = Tkinter.Button(main,text ="Virtual World", command = virtual_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
real  = Tkinter.Button(main,text ="Real World", command = real_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')



launch = Tkinter.Button(top,text ="Launch Simulation", command = launch_sim, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
D = Tkinter.Button(top,text ="Select Bag File", command = launched2, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
G = Tkinter.Button(top,text ="Stop visualization", command = launched0, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
H = Tkinter.Button(top,text ="record bag file", command = launched5, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
I = Tkinter.Button(top,text ="stop recording", command = launched6, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
J = Tkinter.Button(top,text ="stop list", command = launched7, height=1, width=20,fg= '#000000',bg ='#FFFFFF')


virtual.pack()
virtual.place(x=60,y=150)
real.pack()
real.place(x=60,y=250)


launch.pack()
launch.place(x=60,y=150)
D.pack()
D.place(x=60,y=50)
G.pack()
G.place(x=60,y=450)
H.pack()
H.place(x=60,y=250)
I.pack()
I.place(x=60,y=350)
J.pack()
J.place(x=60,y=450)'''

virtual  = Tkinter.Button(main,text ="Virtual World", command = virtual_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
real  = Tkinter.Button(main,text ="Real World", command = real_cmd, height=1, width=20,fg= '#000000',bg ='#FFFFFF')
virtual.pack()
virtual.place(x=60,y=100)
real.pack()
real.place(x=60,y=200)
main.mainloop()
