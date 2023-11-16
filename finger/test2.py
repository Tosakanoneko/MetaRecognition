import serial
from as608_combo_lib import Operation
from PIL import Image
import numpy as np
import as608_combo_lib as as608

session = as608.connect_serial_session("COM5")
if session:
    print(as608.search_fingerprint(session, as608))

# 获取指纹数据
data = session.get_fpdata(sensorbuffer="image")

# 确保 data 是字节类对象
if isinstance(data, list):
    data = bytes(data)

# 打印数据大小
print("Data size:", len(data))

# 处理并保存指纹图像
width, height = 288, 256
image_data = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
image = Image.fromarray(image_data)
image.show()