import os 
import os.path as osp
import sys
import cv2
from pathlib import Path
import shutil
import torch
import math
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image, ImageTk
import tkinter as tk
from tracking_utils.predictor import Predictor
from yolox.utils import fuse_model, get_model_info
from loguru import logger
from tracker.byte_tracker import BYTETracker
from tracking_utils.timer import Timer
from tracking_utils.visualize import plot_tracking, plot_track
from pretreatment import pretreat, imgs2inputs
sys.path.append((os.path.dirname(os.path.abspath(__file__) )) + "/paddle/")
from seg_demo import seg_image
from yolox.exp import get_exp

from font import wrap_text_to_fit_widget

seg_cfgs = {  
    "model":{
        "seg_model" : "./demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml",
    },
    "gait":{
        "dataset": "GREW",
    }
}

def imageflow_demo(video_path, track_result, sil_save_path, message_label, progress, root, canvas):
    """Cuts the video image according to the tracking result to obtain the silhouette

    Args:
        video_path (Path): Path of input video
        track_result (dict): Track information
        sil_save_path (Path): The root directory where the silhouette is stored
    Returns:
        Path: The directory of silhouette
    """
    video_name = os.path.basename(video_path)
    wrapped_text = wrap_text_to_fit_widget(message_label, f"开始分割人物...\n源文件:{video_name}")
    message_label.config(text=wrapped_text)
    root.update()

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_video_name = video_path.split("/")[-1]

    save_video_name = save_video_name.split(".")[0]
    results = []
    ids = list(track_result.keys())
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            if frame_id in ids and frame_id%4==0:
                for tidxywh in track_result[frame_id]:
                    # print('here!!!!!!')
                    # print(tidxywh[0])
                    tid = tidxywh[0]
                    ##############################################
                    if isinstance(tid, int):
                        tidstr = "{:03d}".format(tid)
                    else:
                        tidstr = tid
                    ##############################################
                    savesil_path = osp.join(sil_save_path, save_video_name, tidstr, "undefined")

                    x = tidxywh[1]
                    y = tidxywh[2]
                    width = tidxywh[3]
                    height = tidxywh[4]

                    x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
                    w, h = x2 - x1, y2 - y1
                    x1_new = max(0, int(x1 - 0.1 * w))
                    x2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(x2 + 0.1 * w))
                    y1_new = max(0, int(y1 - 0.1 * h))
                    y2_new = min(int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(y2 + 0.1 * h))
                    
                    new_w = x2_new - x1_new
                    new_h = y2_new - y1_new
                    tmp = frame[y1_new: y2_new, x1_new: x2_new, :]
                    ##############################################
                    tid_int = int(tid) if isinstance(tid, str) else tid
                    frame_id_int = int(frame_id) if isinstance(frame_id, str) else frame_id
                    save_name = "{:03d}-{:03d}.png".format(tid_int, frame_id_int)

                    # save_name = "{:03d}-{:03d}.png".format(tid, frame_id)
                    ##############################################
                    side = max(new_w,new_h)
                    tmp_new = [[[255,255,255]]*side]*side
                    tmp_new = np.array(tmp_new)
                    width = math.floor((side-new_w)/2)
                    height = math.floor((side-new_h)/2)
                    tmp_new[int(height):int(height+new_h),int(width):int(width+new_w),:] = tmp
                    tmp_new = tmp_new.astype(np.uint8)
                    tmp = cv2.resize(tmp_new,(192,192))
                    tmp = seg_image(tmp, seg_cfgs["model"]["seg_model"], save_name, savesil_path)

                    # tmp = tmp.astype(np.uint8)
                    # if tmp.shape[2] == 3 and np.array_equal(tmp[:,:,0], tmp[:,:,1]) and np.array_equal(tmp[:,:,0], tmp[:,:,2]):
                    #     tmp = tmp[:,:,0]
                    # Reshape to remove the single color channel
                    tmp = tmp.reshape(192, 192)

                    # Convert to uint8 data type
                    tmp = tmp.astype(np.uint8)
                    pillow_image = Image.fromarray(tmp).convert('L')
                    
                    # Calculate aspect ratio preserving dimensions
                    canvas_width = canvas.winfo_width()
                    canvas_height = canvas.winfo_height()
                    aspect_ratio = min(canvas_width / tmp.shape[1], canvas_height / tmp.shape[0])
                    new_width = int(tmp.shape[1] * aspect_ratio)
                    new_height = int(tmp.shape[0] * aspect_ratio)

                    # Resize the image using PIL
                    resized_image = pillow_image.resize((new_width, new_height), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(image=resized_image)

                    # Clear previous images and display the new image on the canvas
                    canvas.delete("all")
                    canvas.create_image((canvas_width / 2, canvas_height / 2), image=photo, anchor=tk.CENTER)
                    canvas.image = photo

            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
            
        else:
            break
        frame_id += 1

        # 更新消息标签和进度条
        progress['value'] = (i + 1) / frame_count * 100
        root.update()
    canvas.delete("all")
    return Path(sil_save_path, save_video_name)

def seg(video_path, track_result, sil_save_path, message_label, progress, root, canvas):
    """Cuts the video image according to the tracking result to obtain the silhouette

    Args:
        video_path (Path): Path of input video
        track_result (Path): Track information
        sil_save_path (Path): The root directory where the silhouette is stored
    Returns:
        inputs (list): List of Tuple (seqs, labs, typs, vies, seqL) 
    """
    video_name = os.path.basename(video_path)
    logger.info(f"开始分割{video_name}中人物轮廓")
    sil_save_path = imageflow_demo(video_path, track_result, sil_save_path, message_label, progress, root, canvas)
    inputs = imgs2inputs(Path(sil_save_path), 64, False, seg_cfgs["gait"]["dataset"])
    return inputs

def getsil(video_path, sil_save_path):
    sil_save_name = video_path.split("/")[-1]
    inputs = imgs2inputs(Path(sil_save_path, sil_save_name.split(".")[0]), 
                64, False, seg_cfgs["gait"]["dataset"])
    return inputs
