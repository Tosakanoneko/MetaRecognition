import face_recognition
import cv2

# 加载预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 读取图像
image = cv2.imread('unknown3.jpg')
unknown_image = face_recognition.load_image_file("unknown3.jpg")
# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行人脸检测
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

##############################
# 加载并编码已知人脸
person1_image = face_recognition.load_image_file("person1.jpg")
person1_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("person2.jpg")
person2_encoding = face_recognition.face_encodings(person2_image)[0]

person3_image = face_recognition.load_image_file("person3.jpg")
person3_encoding = face_recognition.face_encodings(person3_image)[0]

# 加载待检测图像

unknown_face_encodings = face_recognition.face_encodings(unknown_image)
# 创建已知人脸编码和名称列表
known_face_encodings = [person1_encoding, person2_encoding, person3_encoding]
known_face_names = ["Person 1", "Person 2", "Person 3"]


# 在未知图像中找到人脸
for unknown_face_encoding in unknown_face_encodings:
    # 比较人脸
    matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding, tolerance=0.5)
    name = "Unknown"

    # 如果找到匹配
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_names[first_match_index]

    print(f"Found {name} in the photo!")

################################
# 在人脸周围画框
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)


# 显示图像
cv2.imshow('Detected Faces', image)

# 等待按键，然后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()


