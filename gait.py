import os
import os.path as osp
import time
import sys
sys.path.append(os.path.abspath('.') + "/demo/libs/")
from track import *
from segment import *
from recognise import *
from clean import *

from PIL import Image, ImageTk
import serial
import shutil
from avi2mp4 import *

import hashlib
import pickle
import os

def compute_hash(filepath):
    """计算文件的哈希值"""
    BUF_SIZE = 65536
    sha1 = hashlib.sha1()
    with open(filepath, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

def gait_recognition(save_root, gallery_video_path, probe_video_path, video_save_folder, message_label, progress, root, canvas):
    # 计算gallery_video的哈希值
    gallery_video_path = gallery_video_path[0]
    gallery_hash = compute_hash(gallery_video_path)
    cache_dir = os.path.join(save_root, 'cache')
    cache_file = os.path.join(cache_dir, f"{gallery_hash}.pkl")

    # 检查是否存在缓存的gallery数据
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            gallery_cache_data = pickle.load(f)
        gallery_track_result = gallery_cache_data['track_result']
        gallery_silhouette = gallery_cache_data['silhouette']
        gallery_feat = gallery_cache_data['feat']
        gallery_track_video_path = gallery_cache_data['gallery_track_video_path']

    else:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        gallery_track_result, gallery_track_video_path = track(gallery_video_path, video_save_folder, message_label, progress, root, canvas)
        gallery_video_name = gallery_video_path.split("/")[-1]
        gallery_video_name = save_root + '/GaitSilhouette/' + gallery_video_name.split(".")[0]
        gallery_silhouette = seg(gallery_video_path, gallery_track_result, save_root + '/GaitSilhouette/', message_label, progress, root,canvas)
        gallery_feat = extract_sil(gallery_silhouette, save_root + '/GaitFeatures/', root)

        with open(cache_file, 'wb') as f:
            pickle.dump({
                'track_result': gallery_track_result,
                'silhouette': gallery_silhouette,
                'feat': gallery_feat,
                'gallery_track_video_path': gallery_track_video_path
            }, f)

    # 对probe_video进行处理
    probe1_track_result, _ = track(probe_video_path, video_save_folder, message_label, progress, root, canvas)
    probe1_video_name = probe_video_path.split("/")[-1]
    probe1_video_name = save_root + '/GaitSilhouette/' + probe1_video_name.split(".")[0]
    probe1_silhouette = seg(probe_video_path, probe1_track_result, save_root + '/GaitSilhouette/', message_label, progress, root, canvas)
    probe1_feat = extract_sil(probe1_silhouette, save_root + '/GaitFeatures/', root)
    gallery_probe1_result, fin_result = compare(probe1_feat, gallery_feat, root)
    # save_video_path = writeresult(gallery_probe1_result, probe_video_path, video_save_folder, message_label, progress, root)
    return gallery_probe1_result, fin_result


