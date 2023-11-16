from pydub import AudioSegment

# 加载WAV文件
sound = AudioSegment.from_wav("recover_test.wav")

# 转换为MP3格式
sound.export("output.mp3", format="mp3")
