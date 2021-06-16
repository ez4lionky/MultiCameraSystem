import math
import time
import yaml
import h5py
import cv2
import threading
import numpy as np
import pyrealsense2 as rs
from itertools import cycle
from threading import Thread
from pathlib import Path
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import pytransform3d.transformations as pt
import pytransform3d.rotations as pr


# webcam frame
class CFrame:
    def __init__(self, frame_number, ts, fps, image, cam_id):
        self.frame_number = frame_number
        self.timestamp = ts
        self.fps = fps
        self.image = image
        self.cam_id = cam_id


# realsense t265 frame
class RSFrame:
    def __init__(self, frame_number, ts, fps, image, pose):
        self.frame_number = frame_number
        self.timestamp = ts
        self.fps = fps
        self.image = image
        self.pose = pose


# Synchronized frame
class SyncFrame:
    def __init__(self, cam_id, frame_number, color_ts, pose_ts, pose_id, cam_pq):
        self.cam_id = cam_id
        self.frame_number = frame_number
        # frame_number in current camera sequence (only including synchronized frames)
        self.color_ts = color_ts
        self.pose_ts = pose_ts
        self.pose_id = pose_id
        self.cam_pq = cam_pq


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

        uptime_s = time.clock_gettime(time.CLOCK_MONOTONIC)
        epoch_s = time.time()
        self.offset_s = epoch_s - uptime_s

        # config and data
        self.save_path = save_path / f'cam{index}_frames'
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        self.save = False
        self.stop = False
        self.frame = None
        self.pre_frame = None
        self.frame_number = 0

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            if self.stop:
                self.capture.release()
                break
            result, img_bgr = self.capture.read()
            local_cam_s = self.capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if not result:
                print(f'{self.cam_id} not available!')
            else:
                time_stamp = local_cam_s + self.offset_s
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if self.frame is None:
                    fps = 0
                    self.frame = CFrame(self.frame_number, time_stamp, fps, img_rgb, self.cam_id)
                    self.pre_frame = self.frame
                else:
                    self.pre_frame = self.frame
                    fps = 1 / (time_stamp - self.pre_frame.timestamp)
                    fps = f'{fps:.3f}'
                    self.frame = CFrame(self.frame_number, time_stamp, fps, img_rgb, self.cam_id)

                self.frame_number += 1
                if self.save:
                    cv2.imwrite(str(self.save_path / f'{time_stamp:.3f}.png'), img_bgr)  # including data transfer time
                # cv2.imwrite(str(self.save_path / f'{st + 1 / 30}.png'), img)  # start_time + 1/30s


class PoseThread(threading.Thread):
    def __init__(self, save_path):
        threading.Thread.__init__(self)
        self.pipe = rs.pipeline()
        # Build config object and request pose data
        self.cfg = rs.config()
        # Start streaming with requested config
        self.pipe.start(self.cfg)

        self.save_path = save_path / 'pose_frames'
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)
        self.pose_data = h5py.File(str(self.save_path / 'PoseFrames.h5'), "a")
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
                self.pipe.stop()
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
                if self.save:
                    cv2.imwrite(str(self.save_path / f'{ts:.3f}.png'), img)
                    if self.pose_data:
                        grp = self.pose_data.create_group(str(ts))
                        tvec = pose.translation
                        rotq = pose.rotation
                        grp.create_dataset('tvec', data=np.array([tvec.x, tvec.y, tvec.z]))
                        grp.create_dataset('rotq', data=np.array([rotq.w, rotq.x, rotq.y, rotq.z]))


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx == 0 or (idx != len(array) and math.fabs(value - array[idx-1]) > math.fabs(value - array[idx])):
        return array[idx], idx
    else:
        return array[idx-1], idx - 1


class MultiCameraSystem:
    def __init__(self, cam_ids, save_path=Path('data'), width=640, height=480, fps=30.0):
        self.cam_num = len(cam_ids)
        self.save_path = save_path
        self.cam_calib = self.read_calib(self)
        self.sync_frames = None

        cam_threads = []
        self.pose_thread = PoseThread(save_path)
        self.width, self.height, self.fps = width, height, fps

        plt.ion()
        self.figure, self.axes, self.sub_imgs = self.visual_cams()
        for i, cam_id in enumerate(cam_ids):
            cam_threads.append(CamThread(i, cam_id, save_path, width, height, fps))
        self.cam_threads = cam_threads
        print("Active threads", threading.activeCount())

        while True:
            init_done = True
            init_done = init_done & (self.pose_thread.rs_frame is not None)
            for cam_thread in cam_threads:
                init_done = init_done & (cam_thread.frame is not None)
            if init_done:
                self.init_all()
                break
            continue

        # 10s
        st = time.time()
        while time.time() - st < 180:
            self.visual_cams()
        self.stop_all_cams()
        self.sync_streams()
        self.analyse_interval()
        return

    @staticmethod
    def read_calib(self, calib_file='cam_calib.yaml'):
        f = open(calib_file, 'r')
        cfg = f.read()
        calib = yaml.safe_load(cfg)
        calib_trans = {}
        for k in calib.keys():
            r_rot = calib[k]['r_rot']
            r_rot = np.array([x / 180 * np.pi for x in r_rot])
            r_trans = np.array(calib[k]['r_trans'])
            calib_trans[k] = pt.transform_from(pr.active_matrix_from_intrinsic_euler_xyz(r_rot), r_trans)
        return calib_trans

    def stop_all_cams(self):
        self.pose_thread.stop = True
        self.pose_thread.pose_data.close()
        for cam_thread in self.cam_threads:
            cam_thread.stop = True

    def init_all(self):
        self.pose_thread.save = True
        for cam_thread in self.cam_threads:
            cam_thread.save = True

    def visual_cams(self):
        cam_num = self.cam_num + 1
        ts = 0
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
                        ts = cur_frame.timestamp
                else:
                    cam_thread = self.cam_threads[i - 1]
                    if cam_thread.frame is not None:
                        cur_frame = cam_thread.frame
                        sub_imgs[i].set_data(cur_frame.image)
                        axes[i].set_title(f'fps: {cur_frame.fps}')
            plt.show()
            plt.pause(0.01)
        return fig, axes, sub_imgs

    def sync_streams(self):
        cam_num = self.cam_num
        pose_file = h5py.File(str(self.save_path / 'pose_frames/PoseFrames.h5'), 'r')
        pose_tss = np.array([float(k) for k in pose_file.keys()])
        pose_tvecs = [np.array(pose_file[str(ts)]['tvec']) for ts in pose_tss]
        pose_rotqs = [np.array(pose_file[str(ts)]['rotq']) for ts in pose_tss]
        multic_tss, multic_tvecs, multic_rotqs = [], [], []

        for i in range(cam_num):
            cur_path = self.save_path / f'cam{i}_frames'
            ts_list = list(cur_path.glob('*.png'))
            ts_list = [float(x.name[:-4]) for x in ts_list]
            multic_tss.append(np.sort(np.array(ts_list)))

        sync_frames = defaultdict(list)
        frame_numbers = []
        for cam_id, curc_tss in enumerate(multic_tss):
            frame_number = 0
            for curc_ts in curc_tss:
                nts, nid = find_nearest(pose_tss * 1000, curc_ts * 1000)
                nts = nts / 1000
                is_sync = math.fabs(curc_ts - nts) * 1000 < 20
                if is_sync:
                    pose = pt.transform_from_pq(np.concatenate([pose_tvecs[nid], pose_rotqs[nid]]))  # xyz wxyz
                    cpose = pt.concat(pose, self.cam_calib[cam_id])
                    cpq = pt.pq_from_transform(cpose)  # x, y, z, qw, qx, qy, qz
                    sframe = SyncFrame(cam_id, frame_number, curc_ts, nts, nid, cpq)
                    sync_frames[cam_id].append(sframe)
                    frame_number += 1
            frame_numbers.append(frame_number)
        self.sync_frames = sync_frames
        self.delay_analysis()
        # self.plot_sys_traj(pose_tvecs, pose_rotqs)
        out_file = h5py.File(str(self.save_path / 'sync_frames.h5'), 'w')
        for k in sync_frames.keys():
            curk_grp = out_file.create_group(str(k))
            for sf in sync_frames[k]:
                grp = curk_grp.create_group(str(sf.frame_number).zfill(6))
                grp.create_dataset('color_ts', data=sf.color_ts)
                grp.create_dataset('pose_ts', data=sf.pose_ts)
                grp.create_dataset('pose_id', data=sf.pose_id)
                grp.create_dataset('cam_pq', data=sf.cam_pq)

    def delay_analysis(self, sync_frames=None, vis=True):
        if sync_frames is None:
            sync_frames = self.sync_frames
        all_pose_tss = []
        delay_dict = defaultdict(list)
        # line_style = {'color': plt.get_cmap('RdYlGn')(np.linspace(0.1, 0.9, len(sync_frames.keys()))),
        #               'marker': ['o', '+', 'x', '*', '.', 'X'], 'linestyle': cycle(["-", "--", "-.", ":"])}
        for k in sync_frames.keys():
            cur_frames = sync_frames[k]
            for sf in cur_frames:
                all_pose_tss.append(sf.pose_ts)
                delay_dict[k].append(math.fabs(sf.pose_ts - sf.color_ts) * 1000)

            # plt.plot(range(len(delay_dict[k])), delay_dict[k], color=line_style['color'][k], linewidth=1,
            #          linestyle=next(line_style['linestyle']), label=f'Camera {k}', marker=line_style['marker'][k])
        if vis:
            plt.figure()
            sns.set(style="darkgrid")
            plt.boxplot(list(delay_dict.values()), labels=[f'Cam {k}' for k in delay_dict.keys()], showmeans=True)
            plt.xlabel('Camera number')
            plt.ylabel('Delay (ms)')
            plt.title('Delay between synchronized frame pair (per color camera and the localization device)')
            plt.show()
            plt.pause(1)

    def plot_sys_traj(self, pose_tvecs, pose_rotqs, sync_frames=None):
        if sync_frames is None:
            sync_frames = self.sync_frames

        all_pose_ids = [sf.pose_id for k in sync_frames.keys() for sf in sync_frames[k]]
        _, uniq_idx, inverse_idx = np.unique(all_pose_ids, return_index=True, return_inverse=True)
        all_sync_frames = list(sync_frames.values())
        for i in uniq_idx:
            cur_pose_id = all_pose_ids[i]
            cur_frame_id = np.where(inverse_idx == cur_pose_id)[0]
            sfs = all_sync_frames[cur_frame_id]
            ref_pose = np.concatenate(pose_tvecs[cur_pose_id], pose_rotqs[cur_pose_id])
            for sf in sfs:
                cur_cam_pose = pt.transform_from_pq(ref_pose)

    def analyse_interval(self):
        frame_num = 20
        frame_buffers = list(self.sync_frames.values())
        init_st = np.min([fs[0].color_ts for fs in frame_buffers])

        st_data = defaultdict(list)
        et_data = defaultdict(list)
        fps_data = defaultdict(list)
        plt.ioff()
        plt.close()
        for cam_id in range(self.cam_num):
            frame_buffer = frame_buffers[cam_id]
            for i in range(1, frame_num + 1):
                if i < len(frame_buffer):
                    pre = frame_buffer[i - 1]
                    cur = frame_buffer[i]
                    st = pre.color_ts - init_st
                    et = cur.color_ts - init_st
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
        category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.1, 0.9, frame_num))
        fig, ax = plt.subplots(figsize=(10.8, self.cam_num))
        plt.title('Timestamp and FPS visualization in 2s interval')
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        ax.invert_yaxis()
        ax.set_xlim(0, np.max(ets))
        ax.set_xlabel('Relative timestamp (second)')
        x_major_locator = plt.MultipleLocator(0.1)
        ax.xaxis.set_major_locator(x_major_locator)
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
                ax.text(x, y, fps_data[y][i], ha='center', va='center', color=text_color)
        plt.show()
        return


if __name__ == '__main__':
    cam_id_list = []
    all_cam_num = 5
    # all_cam_num = 3
    init_id = 2
    for cam_idx in range(all_cam_num):
        cam_id_list.append(f'/dev/video{init_id + cam_idx * 2}')

    mcs = MultiCameraSystem(cam_id_list)
