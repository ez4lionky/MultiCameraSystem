import sys
import math
import time
import h5py
import cv2
import threading
import numpy as np
import pyrealsense2 as rs
from threading import Thread
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class MultiCameraSystem:
    def __init__(self, cam_ids, save_path=Path('data'), width=640, height=480, fps=10.0):
        cam_threads = []
        if not save_path.exists():
            save_path.mkdir()

        # self.pose_thread = PoseThread(save_path)
        self.cam_num = len(cam_ids)
        # self.width, self.height, self.fps = width, height, fps
        # self.save_path = save_path
        #
        # plt.ion()
        # self.figure, self.axes, self.sub_imgs = self.visual_cams()
        # for i, cam_id in enumerate(cam_ids):
        #     cam_threads.append(CamThread(i, cam_id, save_path, width, height, fps))
        # self.cam_threads = cam_threads
        # print("Active threads", threading.activeCount())
        #
        # while True:
        #     init_done = True
        #     init_done = init_done & (self.pose_thread.rs_frame is not None)
        #     for cam_thread in cam_threads:
        #         init_done = init_done & (cam_thread.frame is not None)
        #     if init_done:
        #         self.init_all()
        #         break
        #     # continue
        #
        # # 10s
        # st = time.time()
        # while time.time() - st < 15:
        #     self.visual_cams()
        # self.stop_all_cams()
        self.sync_streams()
        # self.analyse_interval()
        return

    def stop_all_cams(self):
        self.pose_thread.stop = True
        for cam_thread in self.cam_threads:
            cam_thread.stop = True

    def init_all(self):
        self.pose_thread.save = True
        for cam_thread in self.cam_threads:
            cam_thread.save = True

    def visual_cams(self):
        cam_num = self.cam_num + 1
        if not hasattr(self, 'figure'):
            row = 2
            col = int(np.ceil(cam_num / row))
            fig, axes = plt.subplots(row, col, figsize=(30, 40))
            plt.subplots_adjust(left=0.01, bottom=0.01, right=0.98, top=0.98, wspace=0.01, hspace=0.01)
            sub_imgs, init_img = [], np.ones((self.height, self.width, 3))
            axes = np.concatenate(axes)
            for i, ax in enumerate(axes):
                im = ax.imshow(init_img) if i != 0 else ax.imshow(init_img, norm=Normalize(0, 255))
                ax.axis('off')
                sub_imgs.append(im)
        else:
            fig, axes, sub_imgs = self.figure, self.axes, self.sub_imgs
            for i in range(cam_num):
                if i == 0:
                    pose_thread = self.pose_thread
                    if pose_thread.rs_frame is not None:
                        cur_frame = pose_thread.rs_frame
                        sub_imgs[i].set_data(cur_frame.image)
                        axes[i].set_title(f'fps: {cur_frame.fps}')
                else:
                    cam_thread = self.cam_threads[i - 1]
                    if cam_thread.frame is not None:
                        cur_frame = cam_thread.frame
                        sub_imgs[i].set_data(cur_frame.image)
                        axes[i].set_title(f'fps: {1 / (cur_frame.et - cur_frame.st):.2f}')
            plt.show()
            plt.pause(0.01)
        return fig, axes, sub_imgs

    def sync_streams(self):
        cam_num = self.cam_num
        streams_dict = {}.fromkeys(range(cam_num))

    def analyse_interval(self):
        frame_buffers = [cam_thread.frame_buffer for cam_thread in self.cam_threads]
        frame_nums = [len(b) for b in frame_buffers]
        max_frame_num = np.max(frame_nums)
        init_st = np.min([frame_buffer[0].st for frame_buffer in frame_buffers])

        st_data = defaultdict(list)
        et_data = defaultdict(list)
        fps_data = defaultdict(list)
        plt.ioff()
        plt.close()
        # plt.figure()
        for cam_id in range(self.cam_num):
            frame_buffer = frame_buffers[cam_id]
            for i in range(max_frame_num):
                if i < len(frame_buffer):
                    interval = frame_buffer[i]
                    st = interval.st - init_st
                    et = interval.et - init_st
                    fps = f'{1 / (et - st):.1f}'
                else:
                    interval = frame_buffer[-1]
                    st = interval.et - init_st
                    et = st
                    fps = 0
                st_data[cam_id].append(st)
                et_data[cam_id].append(et)
                fps_data[cam_id].append(fps)

        labels = [f'Camera {n}' for n in list(st_data.keys())]
        sts = np.array(list(st_data.values()))
        ets = np.array(list(et_data.values()))
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.1, 0.9, max_frame_num))
        fig, ax = plt.subplots(figsize=(10.8, self.cam_num))
        ax.invert_yaxis()
        # ax.xaxis.set_visible(False)
        ax.set_xlim(0, ets.max())
        for i, color in enumerate(category_colors):
            widths = ets[:, i] - sts[:, i]
            starts = sts[:, i]
            ax.barh(labels, widths, left=starts, label=i, height=0.5, color=color)
            xcenters = starts + widths / 2

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            for y, (x, c) in enumerate(zip(xcenters, widths)):
                if c == 0:
                    continue
                ax.text(x, y, fps_data[y][i], ha='center', va='center',
                        color=text_color)
        plt.show()
        return


class Frame:
    def __init__(self, st, et, image, cam_id):
        self.st = st
        self.et = et
        self.image = image
        self.cam_id = cam_id


class RSFrame:
    def __init__(self, frame_number, ts, fps, image, pose):
        self.frame_number = frame_number
        self.timestamp = ts
        self.fps = fps
        self.image = image
        self.pose = pose


class CamThread(threading.Thread):
    def __init__(self, index, cam_id, save_path, width=640, height=480, fps=10.0):
        threading.Thread.__init__(self)
        self.cam_id = cam_id
        cam = cv2.VideoCapture(cam_id)
        codec = 0x47504A4D  # MJPG
        cam.set(cv2.CAP_PROP_FPS, fps)
        cam.set(cv2.CAP_PROP_FOURCC, codec)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture = cam

        # config and data
        self.save_path = save_path / f'Cam{index}Frames.h5'
        self.data_file = h5py.File(str(self.save_path), "a")

        self.save = False
        self.stop = False
        self.frame = None

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stop:
                break
            st = time.time()
            result, img = self.capture.read()
            et = time.time()
            self.frame = Frame(st, et, img, self.cam_id)
            if not result:
                print(f'{self.cam_id} not available!')
            elif self.save:
                grp = self.data_file.create_group(str(et))
                # grp.create_dataset('st', data=st)
                # grp.create_dataset('et', data=et)
                grp.create_dataset('img', data=img)


class PoseThread(threading.Thread):
    def __init__(self, save_path):
        threading.Thread.__init__(self)
        self.pipe = rs.pipeline()
        # Build config object and request pose data
        self.cfg = rs.config()
        # Start streaming with requested config
        self.pipe.start(self.cfg)

        self.save_path = save_path / 'PoseFrames.h5'
        self.data_file = h5py.File(str(self.save_path), "a")
        self.save = False
        self.stop = False
        self.pre_rs_frame = None
        self.rs_frame = None

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Skip 5 first frames to give the Auto-Exposure time to adjust
        for x in range(5):
            self.pipe.wait_for_frames()

        while True:
            if self.stop:
                break
            frames = self.pipe.wait_for_frames()
            color_stream = frames.get_fisheye_frame()
            img = np.array(color_stream.get_data())
            pose_stream = frames.get_pose_frame()
            pose = pose_stream.get_pose_data()
            frame_number = color_stream.frame_number
            ts = frames.timestamp / 1000

            if self.rs_frame is None:
                fps = 0
                self.rs_frame = RSFrame(frame_number, ts, fps, img, pose)
                self.pre_rs_frame = self.rs_frame
            elif frame_number > self.rs_frame.frame_number:
                self.pre_rs_frame = self.rs_frame
                fps = (frame_number - self.pre_rs_frame.frame_number) / (ts - self.pre_rs_frame.timestamp)
                fps = f'{fps:.3f}'
                self.rs_frame = RSFrame(frame_number, ts, fps, img, pose)
                # print(f'Frame #{frame_number}, fps: {fps}')
                if self.save:
                    grp = self.data_file.create_group(str(ts))
                    tvec = pose.translation
                    rotq = pose.rotation
                    grp.create_dataset('tvec', data=np.array([tvec.x, tvec.y, tvec.z]))
                    grp.create_dataset('rotq', data=np.array([rotq.x, rotq.y, rotq.z, rotq.w]))
                    grp.create_dataset('img', data=img)


if __name__ == '__main__':
    cam_id_list = []
    all_cam_num = 5
    init_id = 2
    for cam_idx in range(all_cam_num):
        cam_id_list.append(f'/dev/video{init_id + cam_idx * 2}')

    mcs = MultiCameraSystem(cam_id_list)
