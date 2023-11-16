import serial
from as608_combo_lib import Operation
from PIL import Image
import numpy as np
import as608_combo_lib 

# 配置串行端口
ser = serial.Serial('COM5', baudrate=57600, timeout=1)

# 创建一个 Operation 类的实例
as608 = Operation(ser)
as608_combo_lib.show_fingerprint_on_device(as608, as608)

# 获取指纹数据
data = as608.get_fpdata(sensorbuffer="image")

# 确保 data 是字节类对象
if isinstance(data, list):
    data = bytes(data)

# 打印数据大小
print("Data size:", len(data))

# 处理并保存指纹图像
width, height = 192, 192  # 假设图像大小为 192x192
image_data = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
image = Image.fromarray(image_data)
image.save("fingerprint.png")
