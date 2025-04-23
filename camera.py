import cv2
import numpy as np
blank_image = np.zeros((100,100,3), np.uint8)

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

while True:
    text_file = open(r"images/data_counter.txt", 'r') 
    search_text = text_file.read()
    img_counter = int(search_text)

    # frame = blank_image
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(str(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))

        img_counter += 1
        with open(r'images/data_counter.txt', 'r') as file: 
            data = file.read()
            data = data.replace(search_text, str(img_counter))
        with open(r'images/data_counter.txt', 'w') as file:
            file.write(data)
        break

cam.release()

cv2.destroyAllWindows()