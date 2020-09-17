import Tkinter as tk
import tkFileDialog
import tkMessageBox
import face_recognition
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL
import numpy as np
import cv2 as cv
import ImageTk
import glob
import os
import pickle
import time

def ocrCallBack():

   root.withdraw()
   global window
   window = tk.Toplevel(root,height=300, width=500, borderwidth=5, bg = '#9bccf7')
   center_window(window)
   window.protocol("WM_DELETE_WINDOW", on_closing)
   window.title("Text Recognition")

   var = tk.IntVar()
   var.set(0)
   pathtext = tk.StringVar()
   pathtext.set(None)

   R1 = tk.Radiobutton(window, text="Use Webcam", variable=var, value=1, bg = '#9bccf7',anchor='w',command=lambda: changeStatus(False,B_browse))
   R1.place(rely = 0.20,relx=0.1,height=35,width=400)
   R2 = tk.Radiobutton(window, text="Select Image", variable=var, value=2, bg = '#9bccf7',anchor='w',command=lambda: changeStatus(True,B_browse))
   R2.place(rely = 0.40,relx=0.1,height=35,width=400)
   B_start = tk.Button(window, text ="Start", borderwidth=3, relief='groove', command=lambda: ocrStartCallBack(var,pathtext),height=2,width=30)
   B_start.place(rely = 0.78,relx=0.35,height=35,width=150)
   B_back = tk.Button(window, text ="Back", borderwidth=3, relief='groove', command=lambda: windowBack(window),height=2,width=30)
   B_back.place(rely = 0.02,relx=0.02,height=28,width=50)
   B_browse = tk.Button(window, text ="Browse", borderwidth=3, relief='groove', command=lambda: browseFile(pathtext),height=2,width=30,state='disabled')
   B_browse.place(rely = 0.55,relx=0.755,height=30,width=80)
   fpath = tk.Label(window,textvariable=pathtext, font=('Helvetica', 11),anchor='w',padx=10 )
   fpath.place(rely = 0.55, relx=0.1,height=30,width=320)

def changeStatus(val,B_browse):
   if val:
       B_browse.config(state="normal")
   else:
       B_browse.config(state="disabled")

def ocrStartCallBack(var,pathtext):
    global img_window
    if var.get() == 0:
        tkMessageBox.showinfo("Info", "Please Select an Option")
        return
    elif var.get() == 1:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            tkMessageBox.showinfo("Info", "Cannot Open Camera")
            return
        ret, frame = cap.read()
        img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        im = Image.fromarray(img)
        im_width, im_height = im.size

        img_window = tk.Toplevel(window, borderwidth=5)
        img_window.bind('<Escape>', lambda e: root.quit())
        center_window(img_window,height=(im_height+ 160),width=(im_width+10))
        imgLabel = tk.Label(img_window)
        imgLabel.pack()
        imgText = tk.Message(img_window, font=('Helvetica', 11),padx=10, anchor='w' )
        imgText.place(rely =((im_height+10)/float(im_height+160)),height=150,width=im_width)
        show_frame(cap,imgLabel,imgText)

    elif var.get() == 2:
        if pathtext.get() == 'None':
            tkMessageBox.showinfo("Info", "Please Select an Image")
            return
        image = Image.open(pathtext.get())
        im_np = np.asarray(image)
        img = cv.cvtColor(im_np, cv.COLOR_RGB2BGR)
        with PyTessBaseAPI() as api:
            api.SetImage(image)
            boxes = api.GetComponentImages(RIL.TEXTLINE, True)
            ocrResult = api.GetUTF8Text()
            #print('Found {} textline image components.'.format(len(boxes)))
            for i, (im, box, _, _) in enumerate(boxes):
                # im is a PIL image object # box is a dict with x, y, w and h keys
                #api.SetRectangle(box['x'], box['y'], box['w'], box['h']) #conf = api.MeanTextConf()
                cv.rectangle(im_np,(box['x'],box['y']),(box['x']+box['w'],box['y']+box['h']),(255,0,0),1)
        im = Image.fromarray(im_np)
        imgtk = ImageTk.PhotoImage(image=im)
        im_width, im_height = im.size
        img_window = tk.Toplevel(window, borderwidth=5)
        img_window.bind('<Escape>', lambda e: root.quit())
        center_window(img_window,height=(im_height+ 30*len(boxes)+10),width=(im_width+10))
        imgLabel = tk.Label(img_window, image=imgtk)
        imgLabel.photo = imgtk
        imgLabel.pack()
        imgText = tk.Text(img_window, font=('Helvetica', 11),padx=10 )
        imgText.insert(tk.INSERT, ocrResult)
        imgText.place(rely =((im_height+10)/float(im_height+30*len(boxes)+10)),height=30*len(boxes),width=im_width)

def windowBack(window):
   root.deiconify()
   window.destroy()

def show_frame(cap,imgLabel,imgText):
    ret, frame = cap.read()
    img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    im = Image.fromarray(img)
    with PyTessBaseAPI() as api:
        api.SetImage(im)
        boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        ocrResult = api.GetUTF8Text()
        #print('Found {} textline image components.'.format(len(boxes)))
        for i, (im, box, _, _) in enumerate(boxes):
        # im is a PIL image object # box is a dict with x, y, w and h keys
        #api.SetRectangle(box['x'], box['y'], box['w'], box['h']) #conf = api.MeanTextConf()
            cv.rectangle(img,(box['x'],box['y']),(box['x']+box['w'],box['y']+box['h']),(255,0,0),1)
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    imgLabel.imgtk = imgtk
    imgLabel.configure(image=imgtk)
    imgText.configure(text=ocrResult)
    imgLabel.after(10, lambda: show_frame(cap,imgLabel,imgText))

def on_closing():
   if tkMessageBox.askokcancel("Quit", "Do you want to quit?"):
      root.destroy()

def fdCallBack():
   root.withdraw()
   global window
   window = tk.Toplevel(root,height=300, width=500, borderwidth=5, bg = '#9bccf7')
   center_window(window)
   window.protocol("WM_DELETE_WINDOW", on_closing)
   window.title("Face Detection")
   var = tk.IntVar()
   var.set(0)
   pathtext = tk.StringVar()
   pathtext.set(None)

   R1 = tk.Radiobutton(window, text="Use Webcam", variable=var, value=1, bg = '#9bccf7',anchor='w',command=lambda: changeStatus(False,B_browse))
   R1.place(rely = 0.20,relx=0.1,height=35,width=400)
   R2 = tk.Radiobutton(window, text="Select Image", variable=var, value=2, bg = '#9bccf7',anchor='w',command=lambda: changeStatus(True,B_browse))
   R2.place(rely = 0.40,relx=0.1,height=35,width=400)
   B_start = tk.Button(window, text ="Start", borderwidth=3, relief='groove', command=lambda: fdStartCallBack(var,pathtext),height=2,width=30)
   B_start.place(rely = 0.78,relx=0.35,height=35,width=150)
   B_back = tk.Button(window, text ="Back", borderwidth=3, relief='groove', command=lambda: windowBack(window),height=2,width=30)
   B_back.place(rely = 0.02,relx=0.02,height=28,width=50)
   B_browse = tk.Button(window, text ="Browse", borderwidth=3, relief='groove', command=lambda: browseFile(pathtext),height=2,width=30,state='disabled')
   B_browse.place(rely = 0.55,relx=0.755,height=30,width=80)
   fpath = tk.Label(window,textvariable=pathtext, font=('Helvetica', 11),anchor='w',padx=10 )
   fpath.place(rely = 0.55, relx=0.1,height=30,width=320)

def fdStartCallBack(var,pathtext):
    global img_window
    global known_face_encodings
    global known_face_names
    if var.get() == 0:
        tkMessageBox.showinfo("Info", "Please Select an Option")
        return
    elif var.get() == 1:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            tkMessageBox.showinfo("Info", "Cannot Open Camera")
            return
        ret, frame = cap.read()
        img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        im = Image.fromarray(img)
        im_width, im_height = im.size

        img_window = tk.Toplevel(window, borderwidth=5)
        img_window.bind('<Escape>', lambda e: root.quit())
        center_window(img_window,height=(im_height+100),width=(im_width+10))
        imgLabel = tk.Label(img_window)
        imgLabel.pack()
        known_face_encodings,known_face_names=loadEncodings()
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        B_add_user = tk.Button(img_window, text ="Add Known Face", borderwidth=3, relief='groove', command=lambda: addUserCallBack(cap,None,False),height=2,width=30)
        B_add_user.place(rely = 0.9,relx=0.40,height=30,width=150)
        show_frame_fd(cap,frame,imgLabel,face_locations,face_encodings,face_names,process_this_frame)

    elif var.get() == 2:
        if pathtext.get() == 'None':
            tkMessageBox.showinfo("Info", "Please Select an Image")
            return
        image = Image.open(pathtext.get())
        im_np = np.asarray(image)
        frame = cv.cvtColor(im_np, cv.COLOR_RGB2BGR)
        frame_orig = cv.cvtColor(im_np, cv.COLOR_RGB2BGR)
        im_height, im_width = frame.shape[:2]
        img_window = tk.Toplevel(window, borderwidth=5)
        img_window.bind('<Escape>', lambda e: root.quit())
        center_window(img_window,height=(im_height+100),width=(im_width+10))
        imgLabel = tk.Label(img_window)
        imgLabel.pack()
        known_face_encodings,known_face_names=loadEncodings()
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        B_add_user = tk.Button(img_window, text ="Add Known Face", borderwidth=3, relief='groove', command=lambda: addUserCallBack(None,frame_orig
,True),height=2,width=30)
        B_add_user.place(rely = (im_height+35)/float(im_height+100),relx=(im_width-140)/float(2*im_width+20),height=30,width=150)
        known_face_encodings,known_face_names=loadEncodings()
        small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                if not face_distances.size == 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                face_names.append(name)
        process_this_frame = not process_this_frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.rectangle(frame, (left, bottom), (right, bottom+20), (0, 0, 255), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(frame, name, (left + 6, bottom+14), font, 0.5, (255, 255, 255), 1)

        img = cv.cvtColor(frame, cv.COLOR_RGB2BGR) 
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        imgLabel.imgtk = imgtk
        imgLabel.configure(image=imgtk)
       
def show_frame_fd(cap,frame,imgLabel,face_locations,face_encodings,face_names,process_this_frame):
    global known_face_encodings
    global known_face_names
    ret, frame = cap.read()
    small_frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
    rgb_small_frame = small_frame[:, :, ::-1]
    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if not face_distances.size == 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
            face_names.append(name)
    process_this_frame = not process_this_frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(frame, (left, bottom), (right, bottom+20), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom+14), font, 0.5, (255, 255, 255), 1)

    img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    im = Image.fromarray(img)       
    imgtk = ImageTk.PhotoImage(image=im)
    imgLabel.imgtk = imgtk
    imgLabel.configure(image=imgtk)
    imgLabel.after(1, lambda: show_frame_fd(cap,frame,imgLabel,face_locations,face_encodings,face_names,process_this_frame))

def generateEncodings():
    
    global known_face_encodings
    global known_face_names
    known_face_encodings = []
    known_face_names = []
    path = '/home/arjun/NTU Project/Face Recognition/data/images/'
    for filename in glob.glob(os.path.join(path,'*.jpg')):
        img = face_recognition.load_image_file(filename)
        if not len(face_recognition.face_encodings(img)) == 0:
            img_encoding = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(img_encoding)
            filename = filename.replace(path,'')
            filename = filename.replace('.jpg','')
            known_face_names.append(filename)
    with open('/home/arjun/NTU Project/Face Recognition/data/enc/encodings.pickle', 'wb') as f:
        pickle.dump([known_face_encodings,known_face_names] , f)

def generateEncodingsName(name):
    
    global known_face_encodings
    global known_face_names
    known_face_encodings = []
    known_face_names = []
    path = '/home/arjun/NTU Project/Face Recognition/data/images/'
    known_face_encodings,known_face_names=loadEncodings()
    img = face_recognition.load_image_file(name)
    if not len(face_recognition.face_encodings(img)) == 0:
        img_encoding = face_recognition.face_encodings(img)[0]
        known_face_encodings.append(img_encoding)
        name = name.replace(path,'')
        name = name.replace('.jpg','')
        known_face_names.append(name)
    with open('/home/arjun/NTU Project/Face Recognition/data/enc/encodings.pickle', 'wb') as f:
        pickle.dump([known_face_encodings,known_face_names] , f)

def loadEncodings():
    filename = '/home/arjun/NTU Project/Face Recognition/data/enc/encodings.pickle'
    with open(filename, 'rb') as f:
        return pickle.load(f)

def addUserCallBack(cap,img,val):
    if val:
        frame = img
    else:
        ret, frame = cap.read()
    global img_window
    img_window.withdraw()
    popup_window = tk.Toplevel(img_window, borderwidth=5)
    img_window.bind('<Escape>', lambda e: img_window.deiconify())
    center_window(popup_window,height=(150),width=(200))
    popup_window.wm_title("Add UnKnown Face")

    l = tk.Label(popup_window, text="Input Name")
    l.place(rely = 0.1, relx=0.05,height=30,width=180)

    E1 = tk.Entry(popup_window)
    E1.place(rely = 0.4, relx=0.20,height=30,width=120)

    b = tk.Button(popup_window, text="Save", command=lambda: exit_popup(frame,E1.get(),popup_window,val))
    b.place(rely = 0.75, relx=0.35,height=30,width=60)

def exit_popup(frame, name,popup_window,val):
    global img_window
    global known_face_encodings
    global known_face_names
    name = '/home/arjun/NTU Project/Face Recognition/data/images/' + name + '.jpg'
    cv.imwrite(name,frame)
    generateEncodingsName(name)
    known_face_encodings,known_face_names = loadEncodings()
    popup_window.destroy()
    if val:
        img_window.destroy()
        var = tk.IntVar()
        var.set(2)
        pathtext = tk.StringVar()
        pathtext.set(name)
        fdStartCallBack(var,pathtext)
    else:
        img_window.deiconify()

def orCallBack():
   root.withdraw()
   global window
   window = tk.Toplevel(root,height=300, width=500, borderwidth=5, bg = '#9bccf7')
   center_window(window)
   window.protocol("WM_DELETE_WINDOW", on_closing)
   window.title("Object Detection")

   var = tk.IntVar()
   var.set(0)
   pathtext = tk.StringVar()
   pathtext.set(None)

   R1 = tk.Radiobutton(window, text="Use Webcam", variable=var, value=1, bg = '#9bccf7',anchor='w',command=lambda: changeStatus(False,B_browse))
   R1.place(rely = 0.20,relx=0.1,height=35,width=400)
   R2 = tk.Radiobutton(window, text="Select Image", variable=var, value=2, bg = '#9bccf7',anchor='w',command=lambda: changeStatus(True,B_browse))
   R2.place(rely = 0.40,relx=0.1,height=35,width=400)
   B_start = tk.Button(window, text ="Start", borderwidth=3, relief='groove', command=lambda: orStartCallBack(var,pathtext),height=2,width=30)
   B_start.place(rely = 0.78,relx=0.35,height=35,width=150)
   B_back = tk.Button(window, text ="Back", borderwidth=3, relief='groove', command=lambda: windowBack(window),height=2,width=30)
   B_back.place(rely = 0.02,relx=0.02,height=28,width=50)
   B_browse = tk.Button(window, text ="Browse", borderwidth=3, relief='groove', command=lambda: browseFile(pathtext),height=2,width=30,state='disabled')
   B_browse.place(rely = 0.55,relx=0.755,height=30,width=80)
   fpath = tk.Label(window,textvariable=pathtext, font=('Helvetica', 11),anchor='w',padx=10 )
   fpath.place(rely = 0.55, relx=0.1,height=30,width=320)

def orStartCallBack(var,pathtext):
    global img_window
    global known_face_encodings
    global known_face_names
    if var.get() == 0:
        tkMessageBox.showinfo("Info", "Please Select an Option")
        return
    elif var.get() == 1:
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            tkMessageBox.showinfo("Info", "Cannot Open Camera")
            return
        ret, frame = cap.read()
        img = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        im = Image.fromarray(img)
        im_width, im_height = im.size
        loadNet()
        img_window = tk.Toplevel(window, borderwidth=5)
        img_window.bind('<Escape>', lambda e: root.quit())
        center_window(img_window,height=(im_height+10),width=(im_width+10))
        imgLabel = tk.Label(img_window)
        imgLabel.pack()
        process_this_frame = True
        show_frame_or(cap,process_this_frame,imgLabel)

    elif var.get() == 2:
        if pathtext.get() == 'None':
            tkMessageBox.showinfo("Info", "Please Select an Image")
            return
        image = Image.open(pathtext.get())
        im_np = np.asarray(image)
        frame = cv.cvtColor(im_np, cv.COLOR_RGB2BGR)
        loadNet()
        frame = detectObjects(frame)
        im_height, im_width = frame.shape[:2]
        img_window = tk.Toplevel(window, borderwidth=5)
        img_window.bind('<Escape>', lambda e: root.quit())
        center_window(img_window,height=(im_height+10),width=(im_width+10))
        imgLabel = tk.Label(img_window)
        imgLabel.pack()
        img = cv.cvtColor(frame, cv.COLOR_RGB2BGR) 
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        imgLabel.imgtk = imgtk
        imgLabel.configure(image=imgtk)

def loadNet():
    global or_net,or_Labels,or_Colors,or_ln
    yolo_path = '/home/arjun/NTU Project/Object Detection/coco trained set/'
    labelsPath = os.path.sep.join([yolo_path, "coco.names"])
    or_Labels = open(labelsPath).read().strip().split("\n")
    np.random.seed(23)
    or_Colors = np.random.randint(0, 255, size=(len(or_Labels), 3)).astype("uint8")
    weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
    configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])
    or_net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
    or_ln = or_net.getLayerNames()
    or_ln = [or_ln[i[0] - 1] for i in or_net.getUnconnectedOutLayers()]

def show_frame_or(cap,process_this_frame,imgLabel):
    for i in range(11):
        cap.read()
    ret, frame = cap.read()
    frame = detectObjects(frame)
    img = cv.cvtColor(frame, cv.COLOR_RGB2BGR) 
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    imgLabel.imgtk = imgtk
    imgLabel.configure(image=imgtk)
    imgLabel.after(1, lambda: show_frame_or(cap,process_this_frame,imgLabel))
   
def detectObjects(image):
    global or_net,or_Labels,or_Colors,or_ln
    min_confidence = 0.5
    nms_threshold = 0.3
    (H, W) = image.shape[:2]
    blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
    or_net.setInput(blob)
    start = time.time()
    layerOutputs = or_net.forward(or_ln)
    end = time.time()


    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
	    for detection in output:
		    scores = detection[5:]
		    classID = np.argmax(scores)
		    confidence = scores[classID]

		    if confidence > min_confidence:
			    box = detection[0:4] * np.array([W, H, W, H])
			    (centerX, centerY, width, height) = box.astype("int")
			    x = int(centerX - (width / 2))
			    y = int(centerY - (height / 2))
			    boxes.append([x, y, int(width), int(height)])
			    confidences.append(float(confidence))
			    classIDs.append(classID)

    idxs = cv.dnn.NMSBoxes(boxes, confidences, min_confidence,nms_threshold)

    if len(idxs) > 0:
	    for i in idxs.flatten():
		    (x, y) = (boxes[i][0], boxes[i][1])
		    (w, h) = (boxes[i][2], boxes[i][3])
		    color = [int(c) for c in or_Colors[classIDs[i]]]
		    cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
		    text = "{}: {:.4f}".format(or_Labels[classIDs[i]], confidences[i])
		    cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    
    return image

def csrCallBack():
   tkMessageBox.showinfo("Info", "Not Yet Ready")

def browseFile(pathtext):
   ofile = tkFileDialog.askopenfilename(parent=root,title='Browse')
   if ofile != None:
      pathtext.set(ofile)

def center_window(root,width=500, height=300):
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    # calculate position x and y coordinates
    x = (screen_width/2) - (width/2)
    y = (screen_height/2) - (height/2)
    root.geometry('%dx%d+%d+%d' % (width, height, x, y))

root = tk.Tk()
root.title("OS Mind")
center_window(root)
root.protocol("WM_DELETE_WINDOW", on_closing)
frame = tk.Frame(root, height=300, width=500, borderwidth=5, bg = '#9bccf7')
frame.pack(expand=False)
frame.pack_propagate(0)
window = None
img_window = None
known_face_encodings = []
known_face_names = []
or_net = []
or_Labels = []
or_Colors = []
or_ln = []

w = tk.Label(frame, text="Select an Option",bg = '#9bccf7', font=('Helvetica', 14))
w.place(rely = 0.02,relx=0.3,height=45,width=200)

B_ocr = tk.Button(frame, text ="Text Recognition", borderwidth=3, relief='groove', command = ocrCallBack,height=2,width=30)
B_ocr.place(rely = 0.2,relx=0.3,height=45,width=200)

B_fd = tk.Button(frame, text ="Face Detection", borderwidth=3, relief='groove', command = fdCallBack,height=2,width=30)
B_fd.place(rely = 0.4,relx=0.3,height=45,width=200)

B_or = tk.Button(frame, text ="Object Detection", borderwidth=3, relief='groove', command = orCallBack,height=2,width=30)
B_or.place(rely = 0.6,relx=0.3,height=45,width=200)

B_csr = tk.Button(frame, text ="Shape Recognition", borderwidth=3, relief='groove', command = csrCallBack,height=2,width=30)
B_csr.place(rely = 0.8,relx=0.3,height=45,width=200)

root.mainloop()


