from scipy.io import wavfile
import noisereduce as nr
import wave
import numpy as np
from pydub import AudioSegment
import os
import win32net

UNC_PATH = r'//192.168.2.99/xilinx/jupyter_notebooks'
unc_path = {
    'remote':UNC_PATH,
    'local':'',
    'username': 'Xilinx',
    'password': 'xilinx'
}

source_pcm_path = os.path.join(UNC_PATH,r'\\192.168.2.99\xilinx\jupyter_notebooks\FPGACompetition\output_pcm.wav')

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

def change_sample_rate(input_file, output_file, new_rate):
    # 加载音频文件
    audio = AudioSegment.from_wav(input_file)
    # 修改采样率并保存
    audio = audio.set_frame_rate(new_rate)
    audio.export(output_file, format="wav")

if __name__ == '__main__':
    win32net.NetUseAdd(None,2,unc_path)
    output_name = "zwbtest"
    # 改变音量
    output_file = './VoiceSrc/output_pcm.wav'
    adjust_volume(source_pcm_path, output_file, 20.0)  # Increase volume by a factor of 2

    # 降噪
    rate, data = wavfile.read('./VoiceSrc/output_pcm.wav')
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write('./VoiceSrc/output_pcm.wav', rate, reduced_noise)

    #修改采样率
    input_path = "./VoiceSrc/output_pcm.wav"
    output_path = "./VoiceSrc/output_pcm.wav"
    desired_sample_rate = 16000  # 例如，将采样率更改为44.1kHz
    change_sample_rate(input_path, output_path, desired_sample_rate)

    # 修改文件类型
    sound = AudioSegment.from_wav('./VoiceSrc/output_pcm.wav')
    sound.export("./VoiceSrc/"+output_name+".mp3", format="mp3")
    win32net.NetUseDel(None, UNC_PATH)  # 关闭UNC口
