import face_recognition
import cv2

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.recent_detections = []
    
    def load_known_faces(self, image_paths, names):
        for img_path, name in zip(image_paths, names):
            img = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(img)[0]
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
    
    def recognize_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        final_result = None

        for (x, y, w, h) in faces:
            face_area = w * h
            screen_area = frame.shape[0] * frame.shape[1]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_ratio = face_area / screen_area
            cv2.putText(frame, f"{face_ratio:.2%}", (frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            if (face_area > screen_area / 200) and (face_area < screen_area / 20):
                face_location = (y, x+w, y+h, x)
                face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
    
                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
    
                self.recent_detections.append(name)
                if len(self.recent_detections) > 3:
                    self.recent_detections.pop(0)
    
                if len(self.recent_detections) == 3 and len(set(self.recent_detections)) == 1:
                    final_result = name
        return frame, final_result


def main():
    face_recognizer = FaceRecognizer()
    face_recognizer.load_known_faces(['person1.jpg', 'person2.jpg', 'person3.jpg'], ["卢泓睿", "郑贝来", "林明杰"])
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        frame, final_result = face_recognizer.recognize_faces(frame)

        if final_result:
            print(f"Final detection result: {final_result}")
            break

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
