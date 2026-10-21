import numpy as np

def apply_affine(affine, pts):
    pts = np.asarray(pts)
    shape = pts.shape
    pts = pts.reshape((-1, 3))
    pts_homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    transformed = (pts_homo @ affine.T)[:, :3]
    return transformed.reshape(shape)
