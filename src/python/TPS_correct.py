import numpy as np
from scipy import ndimage
import cv2
import copy
import find_qr

class TPS():
    def __init__(self):
        self.scale = 1.0
        self.shift = 0.0
    
    def _tps_kernel(self, r):
        r = np.asarray(r)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = (r**2) * np.log(np.where(r == 0, 1.0, r))
        out[r == 0] = 0.0
        return out


    def fit(self, from_pts, to_pts, reg=1e-3):
        """Fit a thin-plate spline mapping ctrl_pts (N,2) -> target_pts (N,2).

        Returns params (N+3,2) and the ctrl_pts (for evaluation).
        params layout: [w (N x 2); a (3 x 2)] stacked vertically.
        """
        ctrl_pts = np.asarray(from_pts, dtype=float)
        target_pts = np.asarray(to_pts, dtype=float)
        if ctrl_pts.ndim != 2 or ctrl_pts.shape[1] != 2:
            raise ValueError('ctrl_pts must be (N,2)')
        N = ctrl_pts.shape[0]
        # Build K
        K = np.zeros((N, N), dtype=float)
        for i in range(N):
            r = np.linalg.norm(ctrl_pts[i] - ctrl_pts, axis=1)
            K[i, :] = self._tps_kernel(r)
        # P matrix (N x 3): [1 x y]
        P = np.column_stack([np.ones(N), ctrl_pts[:, 0], ctrl_pts[:, 1]])
        # Build L matrix
        L = np.zeros((N + 3, N + 3), dtype=float)
        L[:N, :N] = K + reg * np.eye(N)
        L[:N, N:] = P
        L[N:, :N] = P.T
        # Right-hand side
        V = np.vstack([target_pts, np.zeros((3, 2), dtype=float)])
        # Solve
        try:
            self.params = np.linalg.solve(L, V)
        except np.linalg.LinAlgError:
            # fallback to least squares
            self.params, *_ = np.linalg.lstsq(L, V, rcond=None)
        self.ctrl_pts = ctrl_pts

    def tps_transform(self, points):
        """Evaluate TPS at points (M,2). params from fit_tps.

        Returns mapped points (M,2).
        """
        params = np.asarray(self.params, dtype=float)
        ctrl_pts = np.asarray(self.ctrl_pts, dtype=float)
        points = np.asarray(points, dtype=float)
        N = ctrl_pts.shape[0]
        w = params[:N, :]      # (N,2)
        a = params[N:, :]      # (3,2)
        # Pairwise distances: (M,N)
        diff = points[:, None, :] - ctrl_pts[None, :, :]
        r = np.linalg.norm(diff, axis=2)
        U = self._tps_kernel(r)
        mapped = U.dot(w) + np.column_stack([np.ones(points.shape[0]), points]).dot(a)
        mapped = (mapped * self.scale) + self.shift
        # mapped *= 0.5
        return mapped
    
    def affine(self, scale, shift):
        self.scale *= scale
        self.shift += shift


    def warp_image_tps(self, src_img, out_shape, scale = 1, margins = 0, order=1, cval=255):
        """Warp src_img into an output image of shape out_shape (H_out, W_out[, C]).

        src_ctrl_pts: observed landmarks in source image (N,2) as (x,y) coords (col,row)
        dst_ctrl_pts: target (ideal) landmarks in output image coordinates (N,2) as (x,y)

        We fit TPS that maps dst_ctrl_pts -> src_ctrl_pts so that for every output pixel
        (in dst coords) we compute the corresponding source coordinate and sample it.

        order: interpolation order for map_coordinates (0..5)
        cval: constant fill value for out-of-bounds
        reg = 1e-3 … 1e-1
        """
        src = np.asarray(src_img)
        scr = src_img
        h_out, w_out = out_shape[:2]
        # Build grid of destination coordinates (x,y)
        xs = np.arange(w_out)
        ys = np.arange(h_out)
        xv, yv = np.meshgrid(xs, ys)
        grid = np.column_stack([xv.ravel(), yv.ravel()])  # (M,2) with (x,y)
        grid = (grid - margins) / scale
        src_coords = self.tps_transform(grid)
        # src_coords are (x_src, y_src) in image coordinate (col,row)
        x_src = src_coords[:, 0].reshape((h_out, w_out))
        y_src = src_coords[:, 1].reshape((h_out, w_out))
        # Prepare coordinates for map_coordinates: it expects (dim, ...) in row/col order
        coords = np.vstack([y_src.ravel(), x_src.ravel()])
        # If grayscale or single-channel
        if src.ndim == 2:
            warped = ndimage.map_coordinates(src, coords, order=order, mode='constant', cval=cval)
            return warped.reshape((h_out, w_out))
        # Multi-channel
        channels = []
        for ch in range(src.shape[2]):
            band = ndimage.map_coordinates(src[..., ch], coords, order=order, mode='constant', cval=cval)
            channels.append(band.reshape((h_out, w_out)))
        warped = np.stack(channels, axis=-1)
        return warped


def tps_correct(img, tps: TPS, QRsize, square_size = 10):

    margins = square_size * 2

    # tps.affine(square_size, margins)
    
    img_size = QRsize * square_size + margins * 2
    
    wraped = tps.warp_image_tps(img, (img_size, img_size), scale=square_size, margins=margins)

    # tps.affine(1 / square_size, -margins / square_size)

    return wraped

def fit_tps_full_qr(qr, QRsize):
    search = np.asarray([[0, 0], [7, 0], [7, 7], [0, 7]])
    alligment = np.asarray([[0, 0], [3, 0], [3, 3], [0, 3]])
    
    to = np.concatenate([search,
                         search + np.array([QRsize - 7, 0]),
                         alligment + np.array([QRsize - 8, QRsize - 8]),
                         search + np.array([0, QRsize - 7])])
        
    tps = TPS()
    tps.fit(to, qr, reg=1e-3)
    return tps


def fit_tps_alligment_center(qr, QRsize):
    search = np.asarray([[0, 0], [7, 0], [7, 7], [0, 7]])
    alligment = np.asarray([[1.5, 1.5]])
    
    to = np.concatenate([search,
                         search + np.array([QRsize - 7, 0]),
                         alligment + np.array([QRsize - 8, QRsize - 8]),
                         search + np.array([0, QRsize - 7])])
    
    qr = copy.deepcopy(qr)
    qr = np.concatenate([qr[:8], [find_qr.center(qr[8:12])], qr[-4:]])
    
    # print(to)
    
    
    tps = TPS()
    tps.fit(to, qr, reg=1e-3)
    return tps

def fit_tps_no_alligment(qr, QRsize):
    search = np.asarray([[0, 0], [7, 0], [7, 7], [0, 7]])
    alligment = np.asarray([[1.5, 1.5]])
    
    to = np.concatenate([search,
                         search + np.array([QRsize - 7, 0]),
                         search + np.array([0, QRsize - 7])])
    
    qr = copy.deepcopy(qr)
    qr = np.concatenate([qr[:8], qr[-4:]])
    
    # print(to)
    
    
    tps = TPS()
    tps.fit(to, qr, reg=1e-3)
    return tps


if __name__ == "__main__":
    
    # weight, height = img.size
    # scale_ = min(1000 / weight, 1000 / height)
    # img = img.resize((int(weight * scale_), int(height * scale_)))
    
    # qr, img_binarized = find_qr.find_qr_zxing(img)
    
    to = np.asarray([[203, 454], [277, 378], [346, 432], [281, 492], [419, 302], [559, 283], [569, 352], [486, 384], [347, 562], [431, 497], [476, 557], [404, 649]])
    
    fr = np.asarray([[80, 80], [360,  80], [360, 360], [ 80, 360], [640,  80], [920,  80], [920, 360], [640, 360], [ 80, 640], [360, 640], [360, 920], [ 80, 920]])
    
    # qr = np.concatenate([qr[:8], qr[-4:], [find_qr.center(qr[8:12])]])
    
    # wraped = warp_image_tps(img, qr, to, (290, 290), reg=1e-3)
    
    tps = TPS()
    tps.fit(to, fr, reg=1e-3)
    
    img = np.asarray(img)
    
    wraped = tps.warp_image_tps(img, (img.shape[0], img.shape[1]))
    
    cv2.imwrite("corrected.png", wraped)