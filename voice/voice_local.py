import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from scipy.io.wavfile import read
from pydub import AudioSegment
import os
import tkinter as tk
from matplotlib.figure import Figure

import base64
import hashlib
import hmac
import json
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import requests
from scipy.io import wavfile
import noisereduce as nr
import wave

APPId = "911c4613"
APISecret = "ZWFmOWZlMzI4Y2QxY2I5NGUyODJmODZm"
APIKey = "517273ded31c74f43e8efc7b235e8cec"
file_path = './zwbtest.mp3'

groupId="FPGAVoice"
featureId="lmjfeature"
featureInfo="lmjfeatureInfo"

class Gen_req_url(object):
    """生成请求的url"""

    def sha256base64(self, data):
        sha256 = hashlib.sha256()
        sha256.update(data)
        digest = base64.b64encode(sha256.digest()).decode(encoding='utf-8')
        return digest

    def parse_url(self, requset_url):
        stidx = requset_url.index("://")
        host = requset_url[stidx + 3:]
        # self.schema = requset_url[:stidx + 3]
        edidx = host.index("/")
        if edidx <= 0:
            raise Exception("invalid request url:" + requset_url)
        self.path = host[edidx:]
        self.host = host[:edidx]

    # build websocket auth request url
    def assemble_ws_auth_url(self, requset_url, api_key, api_secret, method="GET"):
        self.parse_url(requset_url)
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        # date = "Thu, 12 Dec 2019 01:57:27 GMT"
        signature_origin = "host: {}\ndate: {}\n{} {} HTTP/1.1".format(self.host, date, method, self.path)
        signature_sha = hmac.new(api_secret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        authorization_origin = "api_key=\"%s\", algorithm=\"%s\", headers=\"%s\", signature=\"%s\"" % (
            api_key, "hmac-sha256", "host date request-line", signature_sha)
        authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
        values = {
            "host": self.host,
            "date": date,
            "authorization": authorization
        }

        return requset_url + "?" + urlencode(values)


def gen_req_body(apiname, APPId, file_path=None):
    """
    生成请求的body
    :param apiname
    :param APPId: Appid
    :param file_name:  文件路径
    :return:
    """
    if apiname == 'createFeature':

        with open(file_path, "rb") as f:
            audioBytes = f.read()
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "createFeature",
                    "groupId": groupId,
                    "featureId": featureId,
                    "featureInfo": featureInfo,
                    "createFeatureRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "resource": {
                    "encoding": "lame",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "status": 3,
                    "audio": str(base64.b64encode(audioBytes), 'UTF-8')
                }
            }
        }
    elif apiname == 'createGroup':

        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "createGroup",
                    "groupId": groupId,
                    "groupName": featureId,
                    "groupInfo": featureInfo,
                    "createGroupRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            }
        }
    elif apiname == 'deleteFeature':

        body = {
            "header": {
                "app_id": APPId,
                "status": 3

            },
            "parameter": {
                "s782b4996": {
                    "func": "deleteFeature",
                    "groupId": groupId,
                    "featureId": featureId,
                    "deleteFeatureRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            }
        }
    elif apiname == 'queryFeatureList':

        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "queryFeatureList",
                    "groupId": groupId,
                    "queryFeatureListRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            }
        }
    elif apiname == 'searchFea':

        with open(file_path, "rb") as f:
            audioBytes = f.read()
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "searchFea",
                    "groupId": groupId,
                    "topK": 1,
                    "searchFeaRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "resource": {
                    "encoding": "lame",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "status": 3,
                    "audio": str(base64.b64encode(audioBytes), 'UTF-8')
                }
            }
        }
    elif apiname == 'searchScoreFea':

        with open(file_path, "rb") as f:
            audioBytes = f.read()
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "searchScoreFea",
                    "groupId": groupId,
                    "dstFeatureId": featureId,
                    "searchScoreFeaRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "resource": {
                    "encoding": "lame",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "status": 3,
                    "audio": str(base64.b64encode(audioBytes), 'UTF-8')
                }
            }
        }
    elif apiname == 'updateFeature':

        with open(file_path, "rb") as f:
            audioBytes = f.read()
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "updateFeature",
                    "groupId": groupId,
                    "featureId": featureId,
                    "featureInfo": "iFLYTEK_examples_featureInfo_update",
                    "updateFeatureRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            },
            "payload": {
                "resource": {
                    "encoding": "lame",
                    "sample_rate": 16000,
                    "channels": 1,
                    "bit_depth": 16,
                    "status": 3,
                    "audio": str(base64.b64encode(audioBytes), 'UTF-8')
                }
            }
        }
    elif apiname == 'deleteGroup':
        body = {
            "header": {
                "app_id": APPId,
                "status": 3
            },
            "parameter": {
                "s782b4996": {
                    "func": "deleteGroup",
                    "groupId": groupId,
                    "deleteGroupRes": {
                        "encoding": "utf8",
                        "compress": "raw",
                        "format": "json"
                    }
                }
            }
        }
    else:
        raise Exception(
            "输入的apiname不在[createFeature, createGroup, deleteFeature, queryFeatureList, searchFea, searchScoreFea,updateFeature]内，请检查")
    return body


def find_key_in_nested_dict(d, key_to_search):
    if key_to_search in d:
        return d[key_to_search]

    for key, value in d.items():
        if isinstance(value, dict):
            result = find_key_in_nested_dict(value, key_to_search)
            if result:
                return result

# def req_url(api_name, APPId, APIKey, APISecret, file_path=None):
#     """
#     开始请求
#     :param APPId: APPID
#     :param APIKey:  APIKEY
#     :param APISecret: APISecret
#     :param file_path: body里的文件路径
#     :return:
#     """
#     gen_req_url = Gen_req_url()
#     body = gen_req_body(apiname=api_name, APPId=APPId, file_path=file_path)
#     request_url = gen_req_url.assemble_ws_auth_url(requset_url='https://api.xf-yun.com/v1/private/s782b4996', method="POST", api_key=APIKey, api_secret=APISecret)

#     headers = {'content-type': "application/json", 'host': 'api.xf-yun.com', 'appid': '$APPID'}
#     response = requests.post(request_url, data=json.dumps(body), headers=headers)
#     tempResult = json.loads(response.content.decode('utf-8'))
#     print(tempResult)

#     result = find_key_in_nested_dict(tempResult,'text')
#     feature_id = tempResult["scoreList"][0]["featureId"]

#     print(feature_id)  # 输出: zwbfeature
#     if result:
#         print(str(base64.b64decode(result),'utf-8'))
#     else:
#         print(tempResult)

def req_url(label, api_name, APPId, APIKey, APISecret, file_path=None):
    """
    开始请求
    :param APPId: APPID
    :param APIKey:  APIKEY
    :param APISecret: APISecret
    :param file_path: body里的文件路径
    :return:
    """
    gen_req_url = Gen_req_url()
    body = gen_req_body(apiname=api_name, APPId=APPId, file_path=file_path)
    request_url = gen_req_url.assemble_ws_auth_url(requset_url='https://api.xf-yun.com/v1/private/s782b4996', method="POST", api_key=APIKey, api_secret=APISecret)

    headers = {'content-type': "application/json", 'host': 'api.xf-yun.com', 'appid': APPId}
    response = requests.post(request_url, data=json.dumps(body), headers=headers)
    tempResult = json.loads(response.content.decode('utf-8'))
    # print(tempResult)

    if 'payload' in tempResult and 'searchFeaRes' in tempResult['payload'] and 'text' in tempResult['payload']['searchFeaRes']:
        encoded_text = tempResult['payload']['searchFeaRes']['text']
        decoded_json_str = base64.b64decode(encoded_text).decode('utf-8')
        decoded_dict = json.loads(decoded_json_str)
        
        if 'scoreList' in decoded_dict and decoded_dict['scoreList']:
            feature_id = decoded_dict['scoreList'][0].get('featureId')
            if feature_id is not None:
                print(feature_id)  # 输出: zwbfeature
                if feature_id == "zwbfeature":
                    feature_id = "赵文博"
                    label.config(text=f"人物:{feature_id}", font=("宋体", 13))
                elif feature_id == "lhrfeature":
                    feature_id = "卢泓睿"
                    label.config(text=f"人物:{feature_id}", font=("宋体", 13))
                elif feature_id == "zblfeature":
                    feature_id = "郑贝来"
                    label.config(text=f"人物:{feature_id}", font=("宋体", 13))
                elif feature_id == "lmjfeature":
                    feature_id = "林明杰"
                    label.config(text=f"人物:{feature_id}", font=("宋体", 13))
            else:
                print("Error: 'featureId' not found in scoreList")
        else:
            print("Error: 'scoreList' not found or is empty in decoded text")
    else:
        print("Error: 'text' field not found in response")

def create_spectrum_animation(mp3_path, canvas, audio_running, label):
    def convert_mp3_to_wav(mp3_filename):
        wav_filename = mp3_filename.replace(".mp3", ".wav")
        if not os.path.exists(wav_filename):
            audio = AudioSegment.from_mp3(mp3_filename)
            audio.export(wav_filename, format="wav")
        return wav_filename

    def init():
        line.set_data([], [])
        return (line,)

    def animate(i):
        if not audio_running[0]:  # Stop the animation if audio is not running
            anim.event_source.stop()
            return (line,)
        line.set_data(times[:i*4], data[:i*4])
        return (line,)

    def on_animation_complete():
        print("Animation Complete")
        # audio_running[0] = False
        # The function req_url is not defined in your provided code
        # Make sure to define it or import it before using
        req_url(label, api_name='searchFea', APPId=APPId,
                APIKey=APIKey, APISecret=APISecret, file_path=mp3_path)

    filename = convert_mp3_to_wav(mp3_path)
    samplerate, data = read(filename)
    times = np.arange(len(data)) / float(samplerate)

    downsample_factor = 20
    data = data[::downsample_factor]
    times = times[::downsample_factor]

    fig = Figure()
    ax = fig.add_subplot(111)
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, np.max(times))
    ax.set_ylim(np.min(data), np.max(data))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
    canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(times) // 4, interval=1, blit=True, repeat=False)
    anim._start()

    total_animation_time = int(2.1*len(times))  # frames * interval
    canvas.after(total_animation_time, on_animation_complete)


    return anim

def adjust_volume(input_wav, output_wav, factor):
    """
    Adjust the volume of a WAV file.

    Parameters:
    - input_wav: path to input WAV file.
    - output_wav: path to save the adjusted WAV file.
    - factor: volume adjustment factor. (e.g., 2.0 will double the volume)
    """
    with wave.open(input_wav, 'rb') as wf:
        params = wf.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        audio_data = wf.readframes(nframes)

        # Convert audio to numpy array
        if sampwidth == 1:  # 8-bit samples
            dtype = np.uint8
        elif sampwidth == 2:  # 16-bit samples
            dtype = np.int16
        elif sampwidth == 3:  # 24-bit samples
            dtype = np.int32
            audio_data = audio_data[::3] + audio_data[1::3] + audio_data[2::3]
        else:  # 32-bit samples
            dtype = np.int32

        audio_array = np.frombuffer(audio_data, dtype=dtype)

        # Adjust volume
        audio_array = (audio_array * factor).clip(np.iinfo(dtype).min, np.iinfo(dtype).max).astype(dtype)

        # Save adjusted audio to new WAV file
        with wave.open(output_wav, 'wb') as out_wf:
            out_wf.setparams(params)
            out_wf.writeframes(audio_array.tobytes())

def create_spectrum_image(wav_path, canvas, audio_running, label):
    def convert_mp3_to_wav(mp3_filename):
        wav_filename = mp3_filename.replace(".mp3", ".wav")
        if not os.path.exists(wav_filename):
            audio = AudioSegment.from_mp3(mp3_filename)
            audio.export(wav_filename, format="wav")
        return wav_filename

    adjust_v_wave_path = "./voice_not_recog.wav"
    adjust_volume(wav_path,adjust_v_wave_path,20)

    filename = wav_path
    samplerate, data = read(filename)
    times = np.arange(len(data)) / float(samplerate)

    downsample_factor = 20
    data = data[::downsample_factor]
    times = times[::downsample_factor]

    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot(times, data, lw=2)
    ax.set_xlim(0, np.max(times))
    ax.set_ylim(np.min(data), np.max(data))
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    canvas_widget = FigureCanvasTkAgg(fig, master=canvas)
    canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=1)

    # 降噪
    rate, data = wavfile.read(adjust_v_wave_path)
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(adjust_v_wave_path, rate, reduced_noise)

    # 修改文件类型
    sound = AudioSegment.from_wav(adjust_v_wave_path)
    sound.export("./voice_recog.mp3", format="mp3")
    mp3_path = "./voice_recog.mp3"

    req_url(label, api_name='searchFea', APPId=APPId,
                APIKey=APIKey, APISecret=APISecret, file_path=mp3_path)



# Example usage:
# Create a Tkinter canvas and pass it along with the mp3_path to the function.
# create_spectrum_animation(mp3_path, canvas)
