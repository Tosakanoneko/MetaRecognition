import face_recognition
import cv2

# 加载预训练的人脸检测模型
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载已知的人脸数据
person_images = ['person1.jpg', 'person2.jpg', 'person3.jpg']
known_face_encodings = []
known_face_names = ["Person 1", "Person 2", "Person 3"]

for img_path in person_images:
    img = face_recognition.load_image_file(img_path)
    encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(encoding)

# 获取摄像头
# video_capture = cv2.VideoCapture("C:/Users/25193/Desktop/AutoGait-0.2/OpenGait/demo/output/input2/probe4.mp4")
video_capture = cv2.VideoCapture(1)

recent_detections = []

while True:
    ret, frame = video_capture.read()
    if ret is None:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=9)

    for (x, y, w, h) in faces:
        face_area = w * h
        screen_area = frame.shape[0] * frame.shape[1]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face_ratio = face_area / screen_area
        cv2.putText(frame, f"{face_ratio:.2%}", (frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # # 如果人脸面积超过屏幕的四分之一
        # if face_area > screen_area / 100:
        #     face_location = (y, x+w, y+h, x)
        #     face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
        #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        #     name = "Unknown"

        #     if True in matches:
        #         first_match_index = matches.index(True)
        #         name = known_face_names[first_match_index]
            
        #     # 添加判定结果到列表
        #     recent_detections.append(name)

        #     # 仅保留最近的三个判定结果
        #     if len(recent_detections) > 3:
        #         recent_detections.pop(0)
            
        #     # 检查三次判定结果是否相同
        #     if len(recent_detections) == 3 and len(set(recent_detections)) == 1:
        #         print(f"Final detection result: {name}")
        #         video_capture.release()
        #         cv2.destroyAllWindows()
        #         exit()

    # 显示视频流
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源和关闭窗口
video_capture.release()
cv2.destroyAllWindows()
