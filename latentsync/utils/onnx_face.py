# The face detector (SCRFD, det_10g.onnx) and the 106-point landmark model (2d106det.onnx) of the
# insightface buffalo_l pack, run with onnxruntime alone. The decoding follows insightface 0.7.3
# (model_zoo/retinaface.py and model_zoo/landmark.py, MIT licence). The insightface package has no
# Linux wheel, so a pip install needs a compiler that a runtime image does not have.

import cv2
import numpy as np
import onnxruntime

DET_STRIDES = (8, 16, 32)
DET_ANCHORS = 2
DET_MEAN, DET_STD = 127.5, 128.0
NMS_THRESHOLD = 0.4
# 2d106det.onnx subtracts the mean and scales in its own graph (its first nodes are _minusscalar0 and
# _mulscalar0), so its input is the raw pixel value.
LANDMARK_MEAN, LANDMARK_STD = 0.0, 1.0


def session(path, device):
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device != "cpu" else ["CPUExecutionProvider"]
    available = onnxruntime.get_available_providers()
    return onnxruntime.InferenceSession(path, providers=[p for p in wanted if p in available])


class Face(dict):
    """A detected face: bbox (x1, y1, x2, y2), kps, det_score and landmark_2d_106."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class ScrfdDetector:
    def __init__(self, path, device, input_size, threshold=0.5):
        self.session = session(path, device)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        if len(self.output_names) != 3 * len(DET_STRIDES):
            raise ValueError(f"{path}: expected 9 outputs (scores, boxes, key points), got {len(self.output_names)}")
        self.input_size = tuple(input_size)
        self.threshold = threshold
        self._centers = {}

    def _anchor_centers(self, height, width, stride):
        key = (height, width, stride)
        if key not in self._centers:
            centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (centers * stride).reshape((-1, 2))
            self._centers[key] = np.stack([centers] * DET_ANCHORS, axis=1).reshape((-1, 2))
        return self._centers[key]

    def _forward(self, img):
        blob = cv2.dnn.blobFromImage(
            img, 1.0 / DET_STD, self.input_size, (DET_MEAN, DET_MEAN, DET_MEAN), swapRB=True
        )
        outs = self.session.run(self.output_names, {self.input_name: blob})
        n = len(DET_STRIDES)
        scores, boxes, kpss = [], [], []
        for idx, stride in enumerate(DET_STRIDES):
            centers = self._anchor_centers(blob.shape[2] // stride, blob.shape[3] // stride, stride)
            score = outs[idx].reshape(-1)
            keep = np.where(score >= self.threshold)[0]
            dist = outs[idx + n] * stride
            box = np.stack(
                [centers[:, 0] - dist[:, 0], centers[:, 1] - dist[:, 1],
                 centers[:, 0] + dist[:, 2], centers[:, 1] + dist[:, 3]], axis=-1,
            )
            kps = (outs[idx + 2 * n] * stride).reshape((-1, 5, 2)) + centers[:, None, :]
            scores.append(score[keep])
            boxes.append(box[keep])
            kpss.append(kps[keep])
        return np.concatenate(scores), np.concatenate(boxes), np.concatenate(kpss)

    def detect(self, img):
        """(det, kpss): det rows are x1, y1, x2, y2, score in image pixels."""
        width, height = self.input_size
        if img.shape[0] / img.shape[1] > height / width:
            new_h, new_w = height, int(height * img.shape[1] / img.shape[0])
        else:
            new_w, new_h = width, int(width * img.shape[0] / img.shape[1])
        scale = new_h / img.shape[0]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = cv2.resize(img, (new_w, new_h))
        scores, boxes, kpss = self._forward(canvas)
        order = scores.argsort()[::-1]
        det = np.hstack((boxes / scale, scores[:, None])).astype(np.float32)[order]
        kpss = (kpss / scale)[order]
        keep = nms(det, NMS_THRESHOLD)
        return det[keep], kpss[keep]


def nms(det, threshold):
    x1, y1, x2, y2, scores = det[:, 0], det[:, 1], det[:, 2], det[:, 3], det[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        w = np.maximum(0.0, np.minimum(x2[i], x2[order[1:]]) - np.maximum(x1[i], x1[order[1:]]) + 1)
        h = np.maximum(0.0, np.minimum(y2[i], y2[order[1:]]) - np.maximum(y1[i], y1[order[1:]]) + 1)
        inter = w * h
        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(overlap <= threshold)[0] + 1]
    return keep


class Landmark106:
    def __init__(self, path, device):
        self.session = session(path, device)
        first = self.session.get_inputs()[0]
        self.input_name = first.name
        self.size = int(first.shape[2])
        self.output_names = [o.name for o in self.session.get_outputs()]

    def get(self, img, bbox):
        """106 (x, y) points in image pixels for the face in bbox."""
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        cx, cy = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        scale = self.size / (max(w, h) * 1.5)
        matrix = np.array(
            [[scale, 0.0, self.size / 2 - cx * scale], [0.0, scale, self.size / 2 - cy * scale]], dtype=np.float64
        )
        crop = cv2.warpAffine(img, matrix, (self.size, self.size), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(
            crop, 1.0 / LANDMARK_STD, (self.size, self.size), (LANDMARK_MEAN,) * 3, swapRB=True
        )
        pred = self.session.run(self.output_names, {self.input_name: blob})[0][0].reshape((-1, 2))
        pred = (pred + 1) * (self.size // 2)
        inverse = cv2.invertAffineTransform(matrix)
        return pred @ inverse[:, :2].T + inverse[:, 2]


class FaceModels:
    """Detection plus landmark_2d_106, the two buffalo_l modules that LatentSync uses."""

    def __init__(self, det_path, landmark_path, device, det_size):
        self.detector = ScrfdDetector(det_path, device, det_size)
        self.landmark = Landmark106(landmark_path, device)

    def get(self, img):
        det, kpss = self.detector.detect(img)
        faces = []
        for row, kps in zip(det, kpss):
            face = Face(bbox=row[:4], kps=kps, det_score=float(row[4]))
            face["landmark_2d_106"] = self.landmark.get(img, row[:4])
            faces.append(face)
        return faces
