import numpy as np

def aff2axcodes(affine, labels=(('L', 'R'), ('P', 'A'), ('I', 'S'))):
    """
    Computes axis orientation codes for a 3D affine matrix.
    E.g. ('R', 'A', 'S') for radiological coordinates.
    """
    R = np.asarray(affine[:3, :3], dtype=np.float64)
    codes = []
    for col in range(3):
        col_vec = R[:, col]
        axis = int(np.argmax(np.abs(col_vec)))
        val = col_vec[axis]
        code = labels[axis][1] if val >= 0 else labels[axis][0]
        codes.append(code)
    return tuple(codes)
