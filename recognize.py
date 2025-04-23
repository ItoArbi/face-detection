# Statement for loop
is_true = True

while is_true :
	# import libraries
	import cv2
	import numpy as np

	while True :
		# blank_image = np.zeros((100,100,3), np.uint8)
		cam = cv2.VideoCapture(0)

		ask_first = False
		start_time = 5000

		cv2.namedWindow("test")

		while True:
			# input data how much the the tool had been used for taking pictures
			text_file = open(r"images/data_counter.txt", 'r') 
			search_text = text_file.read()
			img_counter = int(search_text)

			ret, frame = cam.read()
			if not ret:
				print("failed to grab frame")
				break
			cv2.imshow("test", frame)

			start_time -=1

			if start_time < 0 :
				ask_first = True
				break

			k = cv2.waitKey(1)
			if k%256 == 32:
				# SPACE pressed
				img_name = "images/photo_{}.jpg".format(str(img_counter))
				cv2.imwrite(img_name, frame)
				print("{} written!".format(img_name))

				break
		
		cam.release()

		cv2.destroyAllWindows()

		if ask_first == True :
			so = input("\n\n\n\n\n\n\n\n\nNYALAKAN CAMERA LAGI ?   [Y/N]\n:")
			if so == "y" :
				continue
			else :
				break
		else:
			break

	# cam.release()

	# cv2.destroyAllWindows()

	# import the necessary packages
	import numpy as np
	import argparse
	import imutils
	import pickle
	import cv2
	import os
	import tkinter as tk
	root = tk.Tk()

	# specify size of windows.
	root.geometry("495x290")
	root.title("Detail")

	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()
	# ap.add_argument("-i", "--image", required=True,
	# 	help="path to input image")
	ap.add_argument("-d", "--detector", required=True,
		help="path to OpenCV's deep learning face detector")
	ap.add_argument("-m", "--embedding-model", required=True,
		help="path to OpenCV's deep learning face embedding model")
	ap.add_argument("-r", "--recognizer", required=True,
		help="path to model trained to recognize faces")
	ap.add_argument("-l", "--le", required=True,
		help="path to label encoder")
	ap.add_argument("-c", "--confidence", type=float, default=0.5,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())

	# load our serialized face detector from disk
	print("[INFO] loading face detector...")
	protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
	modelPath = os.path.sep.join([args["detector"],
		"res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
	# load our serialized face embedding model from disk
	print("[INFO] loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])
	# load the actual face recognition model along with the label encoder
	recognizer = pickle.loads(open(args["recognizer"], "rb").read())
	le = pickle.loads(open(args["le"], "rb").read())

	text_file = open(r"images/data_counter.txt", 'r') 
	search_text = text_file.read()
	img_counter = int(search_text)

	# load the image, resize it to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image dimensions
	image = cv2.imread("images/photo_{}.jpg".format(search_text))

	img_counter += 1
	with open(r'images/data_counter.txt', 'r') as file: 
		data = file.read()
		data = data.replace(search_text, str(img_counter))
	with open(r'images/data_counter.txt', 'w') as file:
		file.write(data)

	image = imutils.resize(image, width=600)
	(h, w) = image.shape[:2]
	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(image, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)
	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]
		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# extract the face ROI
			face = image[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]
			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
				(0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()
			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			text_data = open("dataset/"+ name +"/doc/data.txt","r")
			raw = text_data.read()
			sample = raw.split(';')

			# input for detail of the user
			age = sample[0]
			relg = sample[1]
			birth = sample[2]
			height = sample[3]
			weight = sample[4]
			addrs = sample[5]
			occ = sample[6]
			full_name = sample[7]

			the_text = "Nama		= "+full_name+"\n\n"+"Umur		= "+age+"\n\n"+"Agama		= "+relg+"\n\n"+"Tanggal Lahir		= "+birth+"\n\n"+"Tinggi Badan		= "+height+"\n\n"+"Berat Badan		= "+weight+"\n\n"+"Alamat		= "+addrs+"\n\n"+"Pekerjaan		= "+occ

			# Create text widget and specify size.
			T = tk.Text(root, height = 15, width = 65)
			T.pack()

			Font_tuple = ("Helvatica", 12, "bold")

			# Insert The Fact.
			T.configure(font = Font_tuple)
			T.insert(tk.END, the_text)

			# draw the bounding box of the face along with the associated
			# probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			cv2.rectangle(image, (startX, startY), (endX, endY),
				(0, 0, 255), 2)
			cv2.putText(image, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			
			text_data.close()

	# show the output image
	cv2.imshow("Image", image)

	tk.mainloop()
	cv2.waitKey(0)

	ask = input("\n\n\n\n\n\n\n\n\n\nNYALAKAN CAMERA LAGI?   [Y / N]\n: ")
	if ask == 'y' or ask == 'Y':
		continue
	elif ask == 'n' or ask == 'N':
		break
	else :
		continue