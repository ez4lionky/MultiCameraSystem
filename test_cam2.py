import cv2
import threading
from threading import Thread


# webcam frame
class CFrame:
    def __init__(self, st, et, image, cam_id):
        self.st = st
        self.et = et
        self.image = image
        self.cam_id = cam_id


class CamThread(threading.Thread):
    def __init__(self, index, cam_id, save_path, width=640, height=480, fps=10.0):
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
        self.save_path = save_path / f'cam{index}_frames'
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        self.save = False
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
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            et = time.time()
            self.frame = CFrame(st, et, img_rgb, self.cam_id)
            if not result:
                print(f'{self.cam_id} not available!')