import base64
with open('fingerprint.bmp', 'rb') as f:
    bmp_data = f.read()
    bmp_b64 = base64.b64encode(bmp_data)
    print(bmp_b64)