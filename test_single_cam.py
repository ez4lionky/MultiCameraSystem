import cv2
import time
import matplotlib.pyplot as plt
import numpy as np


cam = cv2.VideoCapture('/dev/video2')
codec = 0x47504A4D  # MJPG
cam.set(cv2.CAP_PROP_FPS, 30.0)
fps = cam.get(cv2.CAP_PROP_FPS)
print(fps)
cam.set(cv2.CAP_PROP_FOURCC, codec)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fig, axes = plt.subplots(1, 1)
init_img = np.ones((480, 640, 3))
im = axes.imshow(init_img)
axes.axis('off')

result = True
plt.ion()
uptime_s = time.clock_gettime(time.CLOCK_MONOTONIC)
epoch_s = time.time()
offset_s = epoch_s - uptime_s
print('offset_ms', offset_s)
while result:
    st = time.time()
    # result, img_bgr = cam.read()
    cam.grab()
    result, img_bgr = cam.retrieve()
    et = time.time()
    print(st, et)
    if not result:
        print('No image available')
        continue
    fps = f'{1 / (et - st):.1f}'
    local_cam_s = cam.get(cv2.CAP_PROP_POS_MSEC) / 1000
    print('local_cam_s', local_cam_s, 'unix timestamp', local_cam_s + offset_s)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    im.set_data(img_rgb)
    axes.set_title(f'fps: {fps}')
    plt.show()
    plt.pause(0.01)
    # cv2.imwrite(f'test_data/{et}.png', img_bgr)
