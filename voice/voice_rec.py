import base64
import hashlib
import hmac
import json
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from pydub import AudioSegment
import wave
import requests
import numpy as np


# 填写在开放平台申请的APPID、APIKey、APISecret
# 相应编码音频base64编码后数据(不超过4M)
APPId = "911c4613"
APISecret = "ZWFmOWZlMzI4Y2QxY2I5NGUyODJmODZm"
APIKey = "517273ded31c74f43e8efc7b235e8cec"
file_path = './recover_test.wav'

groupId="FPGAVoice"
featureId="lmjfeature"
featureInfo="lmjfeatureInfo"

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
                    feature_id == "郑贝来"
                else:
                    feature_id == "郑贝来"
                label.config(text=f"人物:{feature_id}")
            else:
                print("Error: 'featureId' not found in scoreList")
        else:
            print("Error: 'scoreList' not found or is empty in decoded text")
    else:
        print("Error: 'text' field not found in response")

def change_sample_rate(input_file, output_file, new_rate):
    # 加载音频文件
    audio = AudioSegment.from_wav(input_file)
    # 修改采样率并保存
    audio = audio.set_frame_rate(new_rate)
    audio.export(output_file, format="wav")

def req_url_test(api_name, APPId, APIKey, APISecret, file_path=None):
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
                    feature_id == "郑贝来"
                elif feature_id == "lhrfeature":
                    feature_id == "卢泓睿"
                elif feature_id == "lmjfeature":
                    feature_id == "林明杰"
                print(f"人物:{feature_id}", font=("宋体", 13))
            else:
                print("Error: 'featureId' not found in scoreList")
        else:
            print("Error: 'scoreList' not found or is empty in decoded text")
    else:
        print("Error: 'text' field not found in response")


"""
 * 1.声纹识别接口,请填写在讯飞开放平台-控制台-对应能力页面获取的APPID、APIKey、APISecret
 * 2.groupId要先创建,然后再在createFeature里使用,不然会报错23005,修改时需要注意保持统一
 * 3.音频base64编码后数据(不超过4M),音频格式需要16K、16BIT的MP3音频。
 * 4.主函数只提供调用示例,其他参数请到对应类去更改,以适应实际的应用场景。
"""

if __name__ == '__main__':
    # APPId = "911c4613"
    # APISecret = "ZWFmOWZlMzI4Y2QxY2I5NGUyODJmODZm"
    # APIKey = "517273ded31c74f43e8efc7b235e8cec"
    # file_path = './zwbtest.mp3'

    # groupId="FPGAVoice"
    # featureId="lmjfeature"
    # featureInfo="lmjfeatureInfo"
    
    # apiname取值:
    # 1.创建声纹特征库 createGroup
    # 2.添加音频特征 createFeature
    # 3.查询特征列表 queryFeatureList
    # 4.特征比对1:1 searchScoreFea
    # 5.特征比对1:N searchFea
    # 6.更新音频特征 updateFeature
    # 7.删除指定特征 deleteFeature
    # 8.删除声纹特征库 deleteGroup
    # desired_sample_rate = 44100  # 例如，将采样率更改为44.1kHz

    # # 改变音量
    # output_file = './output_pcm.wav'
    # adjust_volume(output_file, output_file, 20.0)  # Increase volume by a factor of 2

    output_path = "output.wav"
    change_sample_rate(file_path, output_path, 32000)

    # 修改文件类型
    sound = AudioSegment.from_wav(output_path)
    sound.export("./output.mp3", format="mp3")
    file_path = "./output.mp3"
    # audio = AudioSegment.from_mp3(file_path)
    # # 修改采样率并保存
    # audio = audio.set_frame_rate(44100)
    # audio.export("output.mp3", format="mp3")
    req_url_test(api_name='searchFea', APPId=APPId,
            APIKey=APIKey, APISecret=APISecret, file_path=file_path)