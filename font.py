import tkinter as tk
from tkinter import filedialog
import sys
from gait import *
from tkinter import ttk
import textwrap
import tkinter.font as tkfont
from face.face_local import FaceRecognizer
import threading
import cv2 
from PIL import Image, ImageTk
from voice.voice_local import create_spectrum_image
import finger.as608_combo_lib as as608
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import win32net
import os
from playsound import playsound
import zmq
import base64
import numpy as np
import ffmpeg
import queue

def check_queue():
    while not message_queue.empty():
        message = message_queue.get()
        if message == "Popup":
            tk.messagebox.showinfo("身份识别结果", "识别失败，部分模块身份信息不符！")
    root.after(100, check_queue)
message_queue = queue.Queue()

audio_running = [True]
session = as608.connect_serial_session("COM5")

# 初始化 ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://192.168.137.77:5555")
socket.subscribe("")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

UNC_PATH = r'//192.168.2.99/xilinx/jupyter_notebooks'
unc_path = {
    'remote':UNC_PATH,
    'local':'',
    'username': 'Xilinx',
    'password': 'xilinx'
}

# 各模块识别结果初始化
gait_result = ""
face_result = ""
voice_result = ""
finger_result = ""

flag_cam = threading.Event()
flag_voice_end = threading.Event()
flag_voice_start = threading.Event()

win32net.NetUseAdd(None, 2, unc_path)  # 打开UNC口
p2cflag_filepath = os.path.join(UNC_PATH,r'//192.168.2.99/xilinx/jupyter_notebooks/main_code/p2cflag.txt')
c2pflag_filepath = os.path.join(UNC_PATH,r'//192.168.2.99/xilinx/jupyter_notebooks/main_code/c2pflag.txt')
video_path = os.path.join(UNC_PATH,r'//192.168.2.99/xilinx/jupyter_notebooks/main_code/output.avi')
voice_path = os.path.join(UNC_PATH,r'//192.168.2.99/xilinx/jupyter_notebooks/main_code/output_pcm.wav')

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if not event.is_directory and event.src_path == p2cflag_filepath:
            time.sleep(2)
            with open(event.src_path, 'r') as file:
                txt = file.read()
                print("txt更新为：",txt)
                if txt=="cam_ready":
                    flag_cam.set()
                if txt=="voice_start":
                    flag_voice_start.set()
                if txt=="voice_end":
                    flag_voice_end.set()


def start_finger_thread(session, label, fpimg_label, frames, message_queue):
    global audio_label, voice_result, gait_result, face_result, finger_result
    if session:
        as608.search_fingerprint_on_device(session, as608, label, frames, fpimg_label)
        voice_result = audio_label.cget("text")
        finger_result = finger_label.cget("text")

        if gait_result == face_result and gait_result == voice_result and gait_result == finger_result:
            print(f"身份系统识别结果:{gait_result}")
            message_queue.put("Popup")
        else:
            print("身份信息未录入完全或人物不在数据库中!")
            message_queue.put("Popup")

def start_face_thread(label, face_recognizer, face_label, probe_video, frame):
    global face_running, face_result
    face_running = True
    # video_capture = cv2.VideoCapture("./demo/output/input5/1.mp4")
    video_capture = cv2.VideoCapture("./mp4_output.mp4")
    
    frame_width = frame.winfo_width()
    frame_height = frame.winfo_height()
    print("人脸识别启动2！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
    while face_running:
        print("face while run!")
        ret, frame = video_capture.read()
        if not ret:
            continue

        original_frame, final_result = face_recognizer.recognize_faces(frame)
        
        if final_result:
            print("final_result find!!!!!!!!!!!!!!!!!!!!!")
            gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            faces = face_recognizer.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(40, 40),flags=cv2.CASCADE_SCALE_IMAGE)
            
            if len(faces) > 0:
                print("shape find!!!!!!!!!")
                save_path = './saved_frame.jpg'
                cv2.imwrite(save_path, frame)
                (x, y, w, h) = faces[0]

                top = max(0, y - 40)
                left = max(0, x - 40)
                bottom = min(original_frame.shape[0], y + h + 40)
                right = min(original_frame.shape[1], x + w + 40)

                face_frame = original_frame[top:bottom, left:right]
                face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(face_frame)

                # 计算纵横比
                aspect_ratio = img.width / img.height
                new_width = int(min(frame_width, frame_height * aspect_ratio))
                new_height = int(new_width / aspect_ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)

                imgtk = ImageTk.PhotoImage(image=img)
                label.config(image=imgtk)
                label.imgtk = imgtk
                face_label.config(text=f"人物:{final_result}", font=("宋体", 13))
                face_running = False

    face_result = final_result
    video_capture.release()

def wrap_text_to_fit_widget(widget, text):
    # 获取字体信息，以计算每行应该有多少字符
    font = tkfont.Font(font=widget.cget("font"))
    avg_char_width = font.measure("a")  # 假设每个字符的平均宽度接近于字母 "a" 的宽度
    
    widget_char_width = widget.cget("width")
    chars_per_line = widget_char_width

    # 按现有的换行符将文本分解为段落
    paragraphs = text.split("\n")
    
    # 使用textwrap将每个段落分别包装到适当的行数
    wrapped_paragraphs = [textwrap.fill(paragraph, width=chars_per_line) for paragraph in paragraphs]
    
    # 用换行符将它们重新组合
    wrapped_text = "\n".join(wrapped_paragraphs)
    
    return wrapped_text



def select_path(entry):
    file_path = filedialog.askopenfilename(initialdir='./demo/output', filetypes=[("MP4 Files", "*.mp4")])
    if not file_path:
        return
    entry.delete(0, tk.END)
    entry.insert(0, file_path)

def on_close():
    root.destroy()
    sys.exit()

def select_multiple_paths(listbox):
    file_paths = filedialog.askopenfilenames(initialdir='./demo/output', filetypes=[("MP4 Files", "*.mp4")])
    if not file_paths:
        return
    for path in file_paths:
        listbox.insert(tk.END, path)


def start_gait_processing(save_root, video_save_folder):
    from gait import gait_recognition

    global progress, track_message_label, frames, canvases, face_label, predefined_probe_paths, predefined_gallery_paths, gait_result
    gallery_path = predefined_gallery_paths
    probe_paths = predefined_probe_paths
    
    # 第二列:人脸识别
    face_recognizer = FaceRecognizer()
    face_folder = os.path.join(os.path.dirname(__file__), 'face')
    face_recognizer.load_known_faces([os.path.join(face_folder, 'person1.jpg'),
                                  os.path.join(face_folder, 'person2.jpg'),
                                  os.path.join(face_folder, 'person3.jpg')],
                                 ["卢泓睿", "郑贝来", "林明杰"])

    # 视频流标签
    face_video_label = tk.Label(frames[1])
    face_video_label.place(relx=0.5, rely=0.5, anchor='center')

    if probe_paths:
        print("人脸启动！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        # 启动人脸识别线程
        video_thread = threading.Thread(target=start_face_thread, args=(face_video_label, face_recognizer, face_label, predefined_probe_paths[0], frames[1]))
        video_thread.daemon = True
        video_thread.start()

    # 清除frames[4]内的所有组件
    for widget in frames[4].winfo_children():
        widget.destroy()

    # 以下所有的组件都将放在frames[4]内
    track_message_label = tk.Label(frames[4], text=f"", font=("宋体", 13), width=50)
    track_message_label.pack(pady=10)

    def wrap_label_text():
        wrapped_text = wrap_text_to_fit_widget(track_message_label, track_message_label.cget("text"))
        track_message_label.config(text=wrapped_text)

    frames[4].after_idle(wrap_label_text)  # After the main loop starts, this function will be executed

    progress = ttk.Progressbar(frames[4], orient="horizontal", length=500, mode="determinate")
    progress.pack(pady=20)

    frames[4].update()

    # results = []

    for probe_path in probe_paths:
        _, fin_result = gait_recognition(save_root, gallery_path, probe_path, video_save_folder, track_message_label, progress, frames[4], canvases[0])
        # results.append(result)
        # processing_label.config(text=f"处理 {probe_path} 完成!")
        # 清除frames[4]内的所有组件
    gait_final_label = tk.Label(frames[0])
    gait_final_label.place(relx=0.5, rely=0.5, anchor='center')
    for widget in frames[4].winfo_children():
        widget.destroy()
        processing_label = tk.Label(frames[4])
        # processing_label.pack(pady=20, padx=20)
        processing_label.place(relx=0.5, rely=0.5, anchor='center')
    if fin_result == "gallery-001":
        fin_result = "郑贝来"
        gait_final_frame = cv2.imread('C:/Users/25193/Desktop/Gait/MetaGait-1.1/OpenGait/demo/output/GaitSilhouette/mp4_output/001/undefined/001-040.png')
        gait_final_frame = cv2.cvtColor(gait_final_frame, cv2.COLOR_BGR2RGB)
        gait_final_img = Image.fromarray(gait_final_frame)
        gait_final_imgtk = ImageTk.PhotoImage(image=gait_final_img)
        gait_final_label.imgtk = gait_final_imgtk
        gait_final_label.config(image=gait_final_imgtk)
        gait_result = fin_result
        processing_label.config(text=f"人物:{fin_result}")
    elif fin_result == "gallery-002":
        fin_result = "林明杰"
        gait_final_frame = cv2.imread('C:/Users/25193/Desktop/Gait/MetaGait-1.1/OpenGait/demo/output/GaitSilhouette/mp4_output/001/undefined/001-040.png')
        gait_final_frame = cv2.cvtColor(gait_final_frame, cv2.COLOR_BGR2RGB)
        gait_final_img = Image.fromarray(gait_final_frame)
        gait_final_imgtk = ImageTk.PhotoImage(image=gait_final_img)
        gait_final_label.imgtk = gait_final_imgtk
        gait_final_label.config(image=gait_final_imgtk)
        gait_result = fin_result
        processing_label.config(text=f"人物:{fin_result}")
    else:
        processing_label.config(text=f"人物不在库中")
    frames[4].update()

def initial_screen():
    global root
    root = tk.Tk()
    root.title("步态识别")
    root.geometry("1840x840")
    button_font = ('宋体', 20)
    start_button = tk.Button(root, text="开始进行身份识别", command=start_gui, font=button_font, padx=20, pady=10)
    start_button.place(relx=0.5, rely=0.5, anchor='center')

    root.mainloop()

def convert_avi_to_mp4(avi_file_path, mp4_file_path):
    input_stream = ffmpeg.input(avi_file_path)
    ffmpeg.output(input_stream, mp4_file_path).run()

def start_frame_thread():
    global gallery_entry, probe_listbox, root, frames, canvases, face_running, face_label, save_root, video_save_folder, audio_label, finger_label, fpimg_label, voice_result, message_queue
    path = os.path.join(UNC_PATH,r'//192.168.2.99/xilinx/jupyter_notebooks/main_code/')
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

    with open(c2pflag_filepath,'w') as file_w:
        file_w.write("start")

    while True:
        print("wait camera")
        if flag_cam.is_set():
            print("camera ready!")
            playsound('C:/Users/25193/Desktop/Gait/MetaGait-1.1/OpenGait/demo/libs/voice/beep.mp3')
            break
        time.sleep(1)
    
    #网络录制摄像头开始
    print("网络录制摄像头开始")
    start_webcam()
    print("网络录制摄像头结束")
    playsound('C:/Users/25193/Desktop/Gait/MetaGait-1.1/OpenGait/demo/libs/voice/beep.mp3')
    #网络录制摄像头结束
    with open(c2pflag_filepath,'w') as file_w:
        print("写入cam_break")
        file_w.write("cam_break")
    #步态识别代码开始
    print("开始步态识别")
    
    #avi to mp4
    if os.path.exists("./mp4_output.mp4"):
    # 如果文件存在，使用os.remove()来删除它
        os.remove("./mp4_output.mp4")
    convert_avi_to_mp4(video_path,"./mp4_output.mp4")

    gait_thread = threading.Thread(target=start_gait_processing, args=(save_root, video_save_folder))
    gait_thread.daemon = True  # 设置为守护线程
    gait_thread.start()

    #步态识别代码结束

    with open(c2pflag_filepath,'w') as file_w:
        print("写入voice")
        file_w.write("voice")
    time.sleep(4)
    playsound('C:/Users/25193/Desktop/Gait/MetaGait-1.1/OpenGait/demo/libs/voice/beep.mp3')

    while True:
        print("wait voice end")
        if flag_voice_end.is_set():
            print("voice end!")
            break
        time.sleep(1)

    #音频识别代码开始
    print('开始音频识别')

    # 音频部分
    # script_directory = os.path.dirname(__file__)
    
    
    wav_path = voice_path
    # mp3_path  = os.path.join(UNC_PATH,r'\\192.168.2.99\xilinx\jupyter_notebooks\noise\recover_test.wav')
    audio_label.config(text=f"正在等待音频...", font=("宋体", 13))
    
    # 在一个新线程中运行create_spectrum_animation
    audio_thread = threading.Thread(target=create_spectrum_image, args=(wav_path, canvases[2], audio_running, audio_label))
    audio_thread.daemon = True  # 设置为守护线程
    audio_thread.start()

    #音频识别代码结束

    #指纹识别代码开始
    
    finger_thread = threading.Thread(target=start_finger_thread, args=(session, finger_label, fpimg_label, frames[3], message_queue))
    finger_thread.daemon = True  # 设置为守护线程
    finger_thread.start()

    #指纹识别代码结束

    win32net.NetUseDel(None, UNC_PATH)  # 关闭UNC口
    observer.stop()
    observer.join()

    root.after(100, check_queue)

def start_webcam():
    output_path = "C:/Users/25193/Desktop/Gait/MetaGait-1.1/OpenGait/demo/libs/webcam/"
    start_time = time.time()
    # 定义编码器和创建 VideoWriter 对象（用于 MP4 文件）
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
    while True:
        current_time = time.time()
        whole_time = current_time - start_time
        if whole_time >= 6:
            print("结束")
            with open(c2pflag_filepath,'w') as file_w:
                file_w.write("cam_break")
            break
    #     # 接收帧数据
    #     encoded_image = socket.recv()
    
    #     # 解码图像数据
    #     buffer = base64.b64decode(encoded_image)
    #     np_array = np.frombuffer(buffer, dtype=np.uint8)
    #     frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    #     for (x, y, w, h) in faces:
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    #     out.write(frame)
    #     # # 在窗口中显示帧
    #     # cv2.imshow("Receiver", frame)
 
    # # 释放资源
    # out.release()
    # # cv2.destroyAllWindows()

def start_gui():
    global gallery_entry, probe_listbox, root, frames, canvases, face_running, face_label, save_root, video_save_folder, audio_label, finger_label, fpimg_label, predefined_probe_paths, predefined_gallery_paths

    #整体流程线程
    frame_thread = threading.Thread(target=start_frame_thread)
    frame_thread.daemon = True  # 设置为守护线程
    frame_thread.start()

    # 步态处理路径设置
    output_dir = "./demo/output/output4/"
    os.makedirs(output_dir, exist_ok=True)
    current_time = time.localtime()
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    video_save_folder = osp.join(output_dir, timestamp)
    save_root = './demo/output/'

    root.destroy()

    root = tk.Tk()
    root.title("步态识别")
    root.protocol("WM_DELETE_WINDOW", on_close)  # 捕获窗口关闭事件

    # 设置窗口大小为1280x640
    root.geometry("1840x840")

    # 创建8个Frame部件来代表8个区域
    frames = []
    for _ in range(8):
        frame = tk.Frame(root, bd=1, relief='solid')  # 使用bd和relief为每个区域添加边框
        frame.grid(row=_//4, column=_%4, sticky='nsew', padx=5, pady=5)  # 使用grid布局
        frames.append(frame)

    # 为每个frame创建一个Canvas
    canvases = []
    for frame in frames[:4]:  # 只考虑frames[0]至frames[3]
        canvas = tk.Canvas(frame)
        canvas.pack(fill=tk.BOTH, expand=True)
        canvases.append(canvas)

    # 配置行和列的权重，使它们能够按比例放大或缩小
    for i in range(4):
        root.grid_columnconfigure(i, weight=1)
    for j in range(2):
        root.grid_rowconfigure(j, weight=1)

    face_label = tk.Label(frames[5], text="正在等待视频流...", font=("宋体", 13))
    face_label.place(relx=0.5, rely=0.5, anchor='center')

    gait_label = tk.Label(frames[4], text="正在等待视频流...", font=("宋体", 13))
    gait_label.place(relx=0.5, rely=0.5, anchor='center')
    predefined_gallery_paths = ['C:/Users/25193/Desktop/Gait/MetaGait-1.1/OpenGait/demo/output/input4/gallery.mp4']
    # predefined_probe_paths = ["./demo/output/input5/2.mp4"]
    predefined_probe_paths = ["./mp4_output.mp4"]  # 替换为实际的视频文件路径


    audio_label = tk.Label(frames[6], text="正在等待音频...", font=("宋体", 13))
    audio_label.place(relx=0.5, rely=0.5, anchor='center')

    # 指纹部分
    fpimg_label = tk.Label(frames[3])
    fpimg_label.place(relx=0.5, rely=0.5, anchor='center')
    finger_label = tk.Label(frames[7], text="正在等待指纹...", font=("宋体", 13))
    finger_label.place(relx=0.5, rely=0.5, anchor='center')

    root.mainloop()