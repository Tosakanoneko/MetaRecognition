import os
import os.path as osp
import pickle
import sys
import tkinter as tk
import time
# import shutil

root = os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__) )))
sys.path.append(root)
from opengait.utils import config_loader
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))) + "/modeling/")
from loguru import logger
import model.baselineDemo as baselineDemo
import gait_compare as gc

recognise_cfgs = {  
    "gaitmodel":{
        "model_type": "BaselineDemo",
        # "cfg_path": "./configs/baseline/baseline_GREW.yaml",
        "cfg_path": "./configs/gaitbase/gaitbase_da_gait3d.yaml",
    },
}


def loadModel(model_type, cfg_path):
    Model = getattr(baselineDemo, model_type)
    cfgs = config_loader(cfg_path)
    model = Model(cfgs, training=False)
    return model

def gait_sil(sils, embs_save_path):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        embs_save_path (Path): Output path.
    Returns:
        feats (dict): Dictionary of features
    """
    gaitmodel = loadModel(**recognise_cfgs["gaitmodel"])
    gaitmodel.requires_grad_(False)
    gaitmodel.eval()
    feats = {}
    for inputs in sils:
        ipts = gaitmodel.inputs_pretreament(inputs)
        id = inputs[1][0]
        if id not in feats:
            feats[id] = []
        type = inputs[2][0] 
        view = inputs[3][0]
        embs_pkl_path = "{}/{}/{}/{}".format(embs_save_path, id, type, view)
        if not os.path.exists(embs_pkl_path):
            os.makedirs(embs_pkl_path)
        embs_pkl_name = "{}/{}.pkl".format(embs_pkl_path, inputs[3][0])
        retval, embs = gaitmodel.forward(ipts)
        pkl = open(embs_pkl_name, 'wb')
        pickle.dump(embs, pkl)
        feat = {}
        feat[type] = {}
        feat[type][view] = embs
        feats[id].append(feat)        
    return feats

# 直接从pkl文件中提取特征
def rebuild_feats_from_pkls(embs_save_path):
    feats = {}

    for root, dirs, files in os.walk(embs_save_path):
        for filename in files:
            if filename.endswith(".pkl"):
                filepath = os.path.join(root, filename)
                
                parts = root.split(os.path.sep)
                id = parts[-3]
                type = parts[-2]
                view = parts[-1]
                
                with open(filepath, 'rb') as pkl_file:
                    embs = pickle.load(pkl_file)

                if id not in feats:
                    feats[id] = []

                feat = {type: {view: embs}}
                feats[id].append(feat)

    return feats


def gaitfeat_compare(probe_feat:dict, gallery_feat:dict):
    """Compares the feature between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pg_dicts (dict): The id of probe corresponds to the id of gallery
    """
    item = list(probe_feat.keys())
    probe = item[0]
    pg_dict = {}
    pg_dicts = {}
    for inputs in probe_feat[probe]:
        number = list(inputs.keys())[0]
        probeid = probe + "-" + number
        galleryid, idsdict = gc.comparefeat(inputs[number]['undefined'], gallery_feat, probeid, 100)
        pg_dict[probeid] = galleryid
        pg_dicts[probeid] = idsdict
    # print("=================== pg_dicts ===================")
    # print(pg_dicts)
    specific_number = "001"
    key_with_specific_number = next((key for key in pg_dict if specific_number in key), None)

    # 打印结果
    fin_result = pg_dict[key_with_specific_number]
    print(fin_result)
    return pg_dict, fin_result

def extract_sil(sil, save_path, root):
    """Gets the features.

    Args:
        sils (list): List of Tuple (seqs, labs, typs, vies, seqL)
        save_path (Path): Output path.
    Returns:
        video_feats (dict): Dictionary of features from the video
    """
    logger.info("begin extracting")
    for widget in root.winfo_children():
            widget.destroy()
    processing_label = tk.Label(root, text="", font=("宋体", 20))
    processing_label.pack(pady=20, padx=20)
    processing_label.config(text=f"开始提取步态特征")
    root.update()
    time.sleep(1)
    video_feat = gait_sil(sil, save_path)
    processing_label.config(text=f"提取完毕")
    root.update()
    time.sleep(1)
    logger.info("extract Done")
    return video_feat

def rebuild_extract_sil(save_path):
    logger.info("searching gallery features...")
    video_feat = rebuild_feats_from_pkls(save_path)
    return video_feat

def compare(probe_feat, gallery_feat, root):
    """Recognizes  the features between probe and gallery

    Args:
        probe_feat (dict): Dictionary of probe's features
        gallery_feat (dict): Dictionary of gallery's features
    Returns:
        pgdict (dict): The id of probe corresponds to the id of gallery
    """
    logger.info("begin recognising")
    for widget in root.winfo_children():
            widget.destroy()
    processing_label = tk.Label(root, text="", font=("宋体", 20))
    processing_label.pack(pady=20, padx=20)
    processing_label.config(text=f"开始比对步态特征")
    root.update()
    time.sleep(1)
    pgdict, fin_result = gaitfeat_compare(probe_feat, gallery_feat)
    processing_label.config(text=f"比对完毕")
    root.update()
    time.sleep(1)
    logger.info("recognise Done")
    print("================= probe - gallery ===================")
    print(fin_result)
    return pgdict, fin_result