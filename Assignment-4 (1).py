#!/usr/bin/env python
# coding: utf-8

# Assignment-4 (question 2)

# In[6]:


import cv2
import pytesseract
import webbrowser

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
cascPath = "face.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread("qrcode.png",1) 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),   
)

print("Found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), 
                  (0, 0, 255), 2)      
    faces = image[y:y + h, x:x + w]
    cv2.imshow("face",faces)
    cv2.imwrite('face.jpg', faces)
cv2.imwrite('detcted.jpg', image)

ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
 
image2 = image.copy() 
file = open("information.txt", "w+")
file.write("")
file.close()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    rect = cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cropped = image2[y:y + h, x:x + w]
    file = open("information.txt", "a")
    text = pytesseract.image_to_string(cropped)
    file.write(text)
    file.close


detect = cv2.QRCodeDetector()
url_data, bbox, straight_qrcode = detect.detectAndDecode(image)
if url_data:
    webbrowser.open(url_data)


# In[2]:


pip install pytesseract


# In[ ]:




