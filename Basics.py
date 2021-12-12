import cv2
import face_recognition

# getting images
imgElon1 = face_recognition.load_image_file('ImagesBasics/Elon1.jpg')
imgElon2 = face_recognition.load_image_file('ImagesBasics/Elon Musk.jpg')
# converting it to RGB
imgElon1 = cv2.cvtColor(imgElon1, cv2.COLOR_BGR2RGB)
imgElon2 = cv2.cvtColor(imgElon2, cv2.COLOR_BGR2RGB)

# locate face1
faceLoc1 = face_recognition.face_locations(imgElon1)[0]
# encode face
encodeElon1 = face_recognition.face_encodings(imgElon1)[0]
# detection rectangle
cv2.rectangle(imgElon1, (faceLoc1[3], faceLoc1[0]), (faceLoc1[1], faceLoc1[2]), (255, 0, 255), 2)
# print(faceLoc)

# locate face
faceLoc2 = face_recognition.face_locations(imgElon2)[0]
# encode face
encodeElon2 = face_recognition.face_encodings(imgElon2)[0]
# detection rectangle
cv2.rectangle(imgElon2, (faceLoc2[3], faceLoc2[0]), (faceLoc2[1], faceLoc2[2]), (255, 0, 255), 2)
# print(faceLoc)


# comparing
results = face_recognition.compare_faces([encodeElon1], encodeElon2)
faceDistance = face_recognition.face_distance([encodeElon1], encodeElon2)
# print(results , faceDistance)
cv2.putText(imgElon2, f'{results} {round(faceDistance[0], 2)}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

# show image
cv2.imshow('Elon1', imgElon1)
cv2.imshow('Elon2', imgElon2)
cv2.waitKey(0)
