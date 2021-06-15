import cv2
import time
import threading
import numpy as np
from pathlib import Path
from threading import Thread
import matplotlib.pyplot as plt


# webcam frame
class CFrame:
    def __init__(self, st, et, image, cam_id):
        self.st = st
        self.et = et
        self.image = image
        self.cam_id = cam_id


class CamThread(threading.Thread):
    def __init__(self, index, cam_id, save_path, width=640, height=480, fps=30.0):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        cam = cv2.VideoCapture(cam_id)
        codec = 0x47504A4D  # MJPG
        cam.set(cv2.CAP_PROP_FPS, fps)
        # fps = cam.get(cv2.CAP_PROP_FPS)
        cam.set(cv2.CAP_PROP_FOURCC, codec)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture = cam

        # config and data
        self.stop = False
        self.frame = None

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stop:
                self.capture.release()
                break
            st = time.time()
            result, img_bgr = self.capture.read()
            # result = self.capture.grab()
            et = time.time()
            if not result:
                print(f'{self.cam_id} not available!')
            else:
                # result, img_bgr = self.capture.retrieve()
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                self.frame = CFrame(st, et, img_rgb, self.cam_id)


class MultiCameraSystem:
    def __init__(self, cam_ids, save_path=Path('data'), width=640, height=480, fps=30.0):
        self.cam_num = len(cam_ids)
        self.save_path = save_path
        self.sync_frames = None

        cam_threads = []
        self.width, self.height, self.fps = width, height, fps

        plt.ion()
        self.figure, self.axes, self.sub_imgs = self.visual_cams()
        for i, cam_id in enumerate(cam_ids):
            cam_threads.append(CamThread(i, cam_id, save_path, width, height, fps))
        self.cam_threads = cam_threads
        print("Active threads", threading.activeCount())

        while True:
            init_done = True
            for cam_thread in cam_threads:
                init_done = init_done & (cam_thread.frame is not None)
            if init_done:
                break
            continue

        # 10s
        st = time.time()
        while time.time() - st < 120:
            self.visual_cams()
        self.stop_all_cams()
        return

    def stop_all_cams(self):
        for cam_thread in self.cam_threads:
            cam_thread.stop = True

    def visual_cams(self):
        cam_num = self.cam_num
        if not hasattr(self, 'figure'):
            row = 2
            col = int(np.ceil(cam_num / row))
            fig, axes = plt.subplots(row, col, figsize=(30, 40))
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.98, top=0.98, wspace=0.01, hspace=0.01)
            sub_imgs, init_img = [], np.ones((self.height, self.width, 3))
            axes = np.concatenate(axes)
            for i, ax in enumerate(axes):
                im = ax.imshow(init_img) if i != 0 else ax.imshow(init_img)
                ax.axis('off')
                sub_imgs.append(im)
        else:
            fig, axes, sub_imgs = self.figure, self.axes, self.sub_imgs
            for i in range(cam_num):
                cam_thread = self.cam_threads[i - 1]
                if cam_thread.frame is not None:
                    cur_frame = cam_thread.frame
                    sub_imgs[i].set_data(cur_frame.image)
                    axes[i].set_title(f'fps: {1 / (cur_frame.et - cur_frame.st):.2f}')
            plt.show()
            plt.pause(0.01)
        return fig, axes, sub_imgs


cam_id_list = []
all_cam_num = 5
init_id = 2
for cam_idx in range(all_cam_num):
    cam_id_list.append(f'/dev/video{init_id + cam_idx * 2}')

mcs = MultiCameraSystem(cam_id_list)
