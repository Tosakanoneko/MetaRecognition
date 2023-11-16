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
# from seg_demo import seg_image
from font import wrap_text_to_fit_widget
from yolox.exp import get_exp

track_cfgs = {  
    "model":{
        # "seg_model" : "./demo/checkpoints/seg_model/human_pp_humansegv2_mobile_192x192_inference_model_with_softmax/deploy.yaml",
        "ckpt" :    "./demo/checkpoints/bytetrack_model/bytetrack_x_mot17.pth.tar",# 1
        "exp_file": "./demo/checkpoints/bytetrack_model/yolox_x_mix_det.py", # 4
    },
    "gait":{
        "dataset": "GREW",
    },
    "device": "gpu",
    "save_result": "True",
}
colors = [(255,0,0),(0,255,0),(0,0,255),(0,0,0)]
def get_color(idx):
    if idx<=4:
        color = colors[idx-1]
    else:
        idx = idx * 3
        color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def loadckpt(exp):
    device = torch.device("cuda" if track_cfgs["device"] == "gpu" else "cpu")
    model = exp.get_model().to(device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()
    ckpt_file = track_cfgs["model"]["ckpt"]
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    logger.info("\tFusing model...")
    model = fuse_model(model)
    model = model.half()
    return model

exp = get_exp(track_cfgs["model"]["exp_file"], None)
model = loadckpt(exp)

def track(video_path, video_save_folder, message_label, progress, root, canvas):

    """Tracks person in the input video

    Args:
        video_path (Path): Path of input video
        video_save_folder (Path): Tracking video storage root path after processing
    Returns:
        track_results (dict): Track information
    """

    # message_label.config(text=f"开始追踪{video_path}中人物位置")
    video_name = os.path.basename(video_path)
    wrapped_text = wrap_text_to_fit_widget(message_label, f"开始追踪人物位置...\n源文件:{video_name}")
    message_label.config(text=wrapped_text)
    root.update()

    trt_file = None
    decoder = None
    device = torch.device("cuda" if track_cfgs["device"] == "gpu" else "cpu")
    predictor = Predictor(model, exp, trt_file, decoder, device, True)

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = BYTETracker(frame_rate=30)
    timer = Timer()
    frame_id = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(video_save_folder, exist_ok=True)
    save_video_name = video_path.split("/")[-1]
    save_video_path = osp.join(video_save_folder, save_video_name)
    print(f"追踪结果保存路径: {save_video_path}")
    logger.info(f"开始追踪人物位置:{video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

    save_video_name = save_video_name.split(".")[0]
    results = []
    track_results={}
    mark = True
    diff = 0

    # 获取canvas的宽度和高度
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()

        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if mark:
                        mark = False
                        diff = tid - 1
                    # tid = tid - diff
                    tid = "{:03d}".format(tid - diff)
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        if frame_id not in track_results:
                            track_results[frame_id] = []
                        track_results[frame_id].append([tid, tlwh[0], tlwh[1], tlwh[2], tlwh[3]])
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im, ratio = plot_tracking(
                    img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if track_cfgs["save_result"] == "True":
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

            # 更新进度条
            progress_value = (i + 1) / frame_count * 100
            progress['value'] = progress_value
            root.update()
            
            # 将OpenCV的BGR格式图像转换为RGB格式
            rgb_image = cv2.cvtColor(online_im, cv2.COLOR_BGR2RGB)
            # 将图像转换为Pillow的Image对象
            pillow_image = Image.fromarray(rgb_image)
            # 计算等比例缩放的尺寸
            video_aspect_ratio = width / height
            canvas_aspect_ratio = canvas_width / canvas_height

            if video_aspect_ratio > canvas_aspect_ratio:
                new_width = canvas_width
                new_height = int(new_width / video_aspect_ratio)
            else:
                new_height = canvas_height
                new_width = int(new_height * video_aspect_ratio)

            # 使用计算出的新尺寸进行缩放
            resized_image = pillow_image.resize((new_width, new_height), Image.LANCZOS)


            # 清除canvas上的旧图像
            canvas.delete("all")
            # 将缩放后的图像放入canvas
            photo = ImageTk.PhotoImage(image=resized_image)
            canvas.create_image((canvas_width/2, canvas_height/2), image=photo, anchor=tk.CENTER)
            canvas.photo = photo  # keep a reference to avoid garbage collection

            if track_cfgs["save_result"] == "True":
                vid_writer.write(online_im)

            progress_value = (i + 1) / frame_count * 100
            progress['value'] = progress_value
            root.update()
        else:
            break
        

        frame_id += 1
    canvas.delete("all")
    if track_cfgs["save_result"] == "True":
        res_file = osp.join(video_save_folder, f"{save_video_name}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"追踪结果保存至: {res_file}")
    return track_results, save_video_path

def writeresult(pgdict, video_path, video_save_folder, message_label, progress, root):

    """Writes the recognition result back into the video

    Args:
        pgdict (dict): The id of probe corresponds to the id of gallery
        video_path (Path): Path of input video
        video_save_folder (Path): Tracking video storage root path after processing
    """
    video_name = os.path.basename(video_path)
    wrapped_text = wrap_text_to_fit_widget(message_label, f"写入步态识别结果...\n源文件:{video_name}")
    message_label.config(text=wrapped_text)
    root.update()

    device = torch.device("cuda" if track_cfgs["device"] == "gpu" else "cpu")
    trt_file = None
    decoder = None
    predictor = Predictor(model, exp, trt_file, decoder, device, True)
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(video_save_folder, exist_ok=True)
    video_name = video_path.split("/")[-1]
    first_key = next(iter(pgdict))
    if pgdict is not None:
        gallery_name = None
        for key, value in pgdict.items():
            if '001' in key:
                gallery_name = value.split('-')[0]
                break
    else:
        gallery_name = 'gallery'
    probe_name = video_name
    save_video_name = "G-{}_P-{}".format(gallery_name, probe_name)
    # save_video_path = save_video_name.split(".")[0]+ "-After.mp4"
    save_video_path = osp.join(video_save_folder, save_video_name)
    print(f"video save_path is {save_video_path}")
    vid_writer = cv2.VideoWriter(
        save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    video_name = video_name.split(".")[0]

    tracker = BYTETracker(frame_rate=30)
    timer = Timer()
    frame_id = 0
    results = []
    mark = True
    diff = 0
    for i in tqdm(range(frame_count)):
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size)
                online_tlwhs = []
                online_ids = []
                online_colors = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    if mark:
                        mark = False
                        diff = t.track_id - 1
                    track_id = t.track_id - diff

                    pid = "{}-{:03d}".format(video_name, track_id)
                    tid = pgdict[pid]
                    # demo
                    colorid = int(tid.split("-")[1])
                    # colorid = track_id
                    vertical = tlwh[2] / tlwh[3] > 1.6
                    if tlwh[2] * tlwh[3] > 10 and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_colors.append(colorid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_track(
                    img_info['raw_img'], online_tlwhs, online_ids, online_colors, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_img']
            if track_cfgs["save_result"] == "True":
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

        progress['value'] = (i + 1) / frame_count * 100
        root.update()
    
    return save_video_path

    

    # if track_cfgs["save_result"] == "True":
    #     txtfile = "{}-{}".format(save_video_name, "After.txt")
    #     res_file = osp.join(video_save_folder, txtfile)
    #     with open(res_file, 'w') as f:
    #         f.writelines(results)
    #     logger.info(f"save results to {res_file}")
