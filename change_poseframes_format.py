import h5py
import numpy as np

pose_file = h5py.File('data/pose_frames/PoseFrames.h5', 'r')
pose_tss = np.array([float(k) for k in pose_file.keys()])
pose_tvecs = [np.array(pose_file[str(ts)]['tvec']) for ts in pose_tss]
pose_rotqs = [np.array(pose_file[str(ts)]['rotq']) for ts in pose_tss]

new_file = h5py.File('data/pose_frames/new_PoseFrames.h5', 'w')
for i, ts in enumerate(pose_tss):
    grp = new_file.create_group(str(ts))
    tvec = pose_tvecs[i]
    rotq = pose_rotqs[i]
    grp.create_dataset('tvec', data=np.array(tvec))
    grp.create_dataset('rotq', data=np.array([rotq[3], rotq[0], rotq[1], rotq[2]]))
