from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
import os

import cv2
import numpy as np


RAW_HAND_LANDMARK_DIM = 68
HAND_PRESENCE_INDEX = RAW_HAND_LANDMARK_DIM - 1
HAND_POSE_FEATURE_DIM = 48
HAND_LANDMARK_DIM = RAW_HAND_LANDMARK_DIM + HAND_POSE_FEATURE_DIM


@dataclass
class HandLandmarkResult:
    features: np.ndarray
    presence: float
    landmarks: np.ndarray | None = None


class HandLandmarkExtractor:
    """
    Extract one-hand palm and finger-joint landmarks.

    Feature layout:
        21 landmarks * (x, y, z) = 63
        bounding box x1, y1, x2, y2 = 4
        hand presence = 1
        finger pose features = 48
        total = 116
    """

    def __init__(
        self,
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.35,
        model_asset_path=None,
        allow_contour_fallback=False,
    ):
        self.available = False
        self.uses_contour_fallback = False
        self.allow_contour_fallback = allow_contour_fallback
        self._hands = None
        self._task_landmarker = None
        self._mp_image = None
        self._mp_image_format = None
        self.backend = "unavailable"

        try:
            import mediapipe as mp
            self._mp_hands = getattr(getattr(mp, "solutions", None), "hands", None)
        except (ImportError, AttributeError):
            self._mp_hands = None

        if self._mp_hands is not None:
            self.available = True
            self.backend = "solutions"
            self._hands = self._mp_hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=0.35,
            )
            return

        task_path = self._resolve_task_model(model_asset_path)
        if task_path is not None and self._init_task_landmarker(
            task_path,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
        ):
            return

        message = (
            "True MediaPipe hand landmarks are not configured. Place "
            "hand_landmarker.task in models/ or assets/, or set HAND_LANDMARKER_TASK "
            "to the model file path. Contour fallback is disabled because it is not "
            "reliable enough for this app."
        )
        if not allow_contour_fallback:
            raise RuntimeError(message)

        warnings.warn(
            message + " Using contour fallback only because allow_contour_fallback=True.",
            RuntimeWarning,
        )
        self.uses_contour_fallback = True
        self.backend = "contour"

    @staticmethod
    def _resolve_task_model(model_asset_path):
        candidates = []
        if model_asset_path:
            candidates.append(Path(model_asset_path))
        env_path = os.environ.get("HAND_LANDMARKER_TASK")
        if env_path:
            candidates.append(Path(env_path))
        candidates.extend([
            Path("models/hand_landmarker.task"),
            Path("assets/hand_landmarker.task"),
        ])

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _init_task_landmarker(self, task_path, max_num_hands, min_detection_confidence):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(task_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=max_num_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=0.35,
                min_tracking_confidence=0.35,
            )
            self._task_landmarker = vision.HandLandmarker.create_from_options(options)
            self._mp_image = mp.Image
            self._mp_image_format = mp.ImageFormat.SRGB
            self.available = True
            self.backend = "tasks"
            return True
        except Exception as exc:
            warnings.warn(
                f"Could not initialize MediaPipe HandLandmarker from {task_path}: {exc}. "
                "Contour fallback remains disabled unless allow_contour_fallback=True.",
                RuntimeWarning,
            )
            return False

    @staticmethod
    def empty() -> np.ndarray:
        return np.zeros(HAND_LANDMARK_DIM, dtype=np.float32)

    @classmethod
    def features_from_landmarks(cls, coords, presence=1.0) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float32)
        if coords.shape != (21, 3):
            return cls.empty()

        wrist = coords[0, :2].copy()
        palm_size = np.linalg.norm(coords[9, :2] - coords[0, :2])
        scale = max(float(palm_size), 1e-3)

        normalized = coords.copy()
        normalized[:, 0] = (normalized[:, 0] - wrist[0]) / scale
        normalized[:, 1] = (normalized[:, 1] - wrist[1]) / scale
        normalized[:, 2] = normalized[:, 2] / scale

        x1, y1 = coords[:, :2].min(axis=0)
        x2, y2 = coords[:, :2].max(axis=0)
        bbox = np.array([x1, y1, x2, y2], dtype=np.float32)
        return cls._build_feature_vector(normalized, bbox, presence).astype(np.float32)

    @staticmethod
    def crop_to_landmarks(image_rgb, coords, padding=0.35):
        """
        Crop around true hand landmarks and remap landmarks into crop coordinates.

        Returns (crop_rgb, crop_coords, bounds). If landmarks are missing, returns
        the original image, None, and the full-image bounds.
        """
        image = HandLandmarkExtractor._to_uint8_rgb(image_rgb)
        h, w = image.shape[:2]
        full_bounds = (0, 0, w, h)
        if coords is None:
            return image, None, full_bounds

        coords = np.asarray(coords, dtype=np.float32)
        if coords.shape != (21, 3):
            return image, None, full_bounds

        xs = coords[:, 0] * float(w)
        ys = coords[:, 1] * float(h)
        x1 = float(xs.min())
        y1 = float(ys.min())
        x2 = float(xs.max())
        y2 = float(ys.max())
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        pad = padding * max(box_w, box_h)

        left = max(0, int(np.floor(x1 - pad)))
        top = max(0, int(np.floor(y1 - pad)))
        right = min(w, int(np.ceil(x2 + pad)))
        bottom = min(h, int(np.ceil(y2 + pad)))
        if right <= left or bottom <= top:
            return image, None, full_bounds

        crop = image[top:bottom, left:right]
        crop_w = max(1.0, float(right - left))
        crop_h = max(1.0, float(bottom - top))
        crop_coords = coords.copy()
        crop_coords[:, 0] = (xs - float(left)) / crop_w
        crop_coords[:, 1] = (ys - float(top)) / crop_h
        crop_coords[:, 0] = np.clip(crop_coords[:, 0], 0.0, 1.0)
        crop_coords[:, 1] = np.clip(crop_coords[:, 1], 0.0, 1.0)
        return crop, crop_coords.astype(np.float32), (left, top, right, bottom)

    def extract(self, image_rgb) -> HandLandmarkResult:
        if self.backend == "tasks" and self._task_landmarker is not None:
            return self._extract_with_tasks(image_rgb)

        if not self.available or self._hands is None:
            if not self.allow_contour_fallback:
                return HandLandmarkResult(self.empty(), 0.0, None)
            return self._extract_contour_fallback(image_rgb)

        image = self._to_uint8_rgb(image_rgb)
        results = self._hands.process(image)
        if not results.multi_hand_landmarks:
            return HandLandmarkResult(self.empty(), 0.0, None)

        hand = results.multi_hand_landmarks[0]
        coords = np.array(
            [[landmark.x, landmark.y, landmark.z] for landmark in hand.landmark],
            dtype=np.float32,
        )

        features = self.features_from_landmarks(coords, 1.0)
        return HandLandmarkResult(features.astype(np.float32), 1.0, coords.astype(np.float32))

    def _extract_with_tasks(self, image_rgb) -> HandLandmarkResult:
        image = self._to_uint8_rgb(image_rgb)
        mp_image = self._mp_image(image_format=self._mp_image_format, data=image)
        result = self._task_landmarker.detect(mp_image)
        if not result.hand_landmarks:
            return HandLandmarkResult(self.empty(), 0.0, None)

        hand = result.hand_landmarks[0]
        coords = np.array([[point.x, point.y, point.z] for point in hand], dtype=np.float32)
        presence = 1.0
        if result.handedness and result.handedness[0]:
            presence = float(result.handedness[0][0].score)

        features = self.features_from_landmarks(coords, presence)
        return HandLandmarkResult(features.astype(np.float32), presence, coords.astype(np.float32))

    def _extract_contour_fallback(self, image_rgb) -> HandLandmarkResult:
        image = self._to_uint8_rgb(image_rgb)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(
            ycrcb,
            np.array([10, 120, 55], dtype=np.uint8),
            np.array([255, 190, 155], dtype=np.uint8),
        )

        mean, std = cv2.meanStdDev(gray)
        threshold = max(18.0, float(mean[0][0] + 0.45 * std[0][0]))
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_or(skin_mask, bright_mask)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return HandLandmarkResult(self.empty(), 0.0, None)

        contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(contour) / float(mask.size)
        if area_ratio < 0.01:
            return HandLandmarkResult(self.empty(), 0.0, None)

        points = contour.reshape(-1, 2).astype(np.float32)
        sampled = self._sample_points(points, 20)
        center = points.mean(axis=0, keepdims=True)
        coords_2d = np.concatenate([center, sampled], axis=0)

        h, w = image.shape[:2]
        coords = np.zeros((21, 3), dtype=np.float32)
        coords[:, 0] = coords_2d[:, 0] / max(float(w), 1.0)
        coords[:, 1] = coords_2d[:, 1] / max(float(h), 1.0)

        wrist = coords[0, :2].copy()
        palm_size = np.sqrt(area_ratio)
        scale = max(float(palm_size), 1e-3)

        normalized = coords.copy()
        normalized[:, 0] = (normalized[:, 0] - wrist[0]) / scale
        normalized[:, 1] = (normalized[:, 1] - wrist[1]) / scale

        x, y, bw, bh = cv2.boundingRect(contour)
        bbox = np.array([x / w, y / h, (x + bw) / w, (y + bh) / h], dtype=np.float32)
        features = self._build_feature_vector(normalized, bbox, 1.0)
        return HandLandmarkResult(
            features.astype(np.float32),
            float(min(1.0, area_ratio / 0.08)),
            coords.astype(np.float32),
        )

    @classmethod
    def _build_feature_vector(cls, normalized, bbox, presence):
        pose = cls._finger_pose_features(normalized)
        return np.concatenate(
            [
                normalized.reshape(-1),
                bbox.astype(np.float32),
                np.array([presence], dtype=np.float32),
                pose,
            ]
        )

    @staticmethod
    def _safe_norm(vector):
        return max(float(np.linalg.norm(vector)), 1e-6)

    @classmethod
    def _unit_vector(cls, vector):
        return vector / cls._safe_norm(vector)

    @classmethod
    def _joint_angle(cls, a, b, c):
        ba = a - b
        bc = c - b
        denom = cls._safe_norm(ba) * cls._safe_norm(bc)
        cosine = float(np.dot(ba, bc) / denom)
        return np.float32(np.clip(cosine, -1.0, 1.0))

    @classmethod
    def _finger_pose_features(cls, normalized):
        """
        Finger-aware pose features derived from MediaPipe's 21 hand joints.

        Per finger:
            tip distance from wrist
            tip distance from MCP/base joint
            extension ratio: straight distance / bone-chain length
            two joint angle cosines
            MCP/base-to-tip unit direction x, y, z

        Cross-finger:
            adjacent fingertip distances
            adjacent finger direction angle cosines
        """
        coords = normalized.astype(np.float32)
        wrist = coords[0]
        fingers = [
            (1, 2, 3, 4),      # thumb
            (5, 6, 7, 8),      # index
            (9, 10, 11, 12),   # middle
            (13, 14, 15, 16),  # ring
            (17, 18, 19, 20),  # pinky
        ]

        features = []
        directions = []
        tips = []
        for joints in fingers:
            pts = coords[list(joints)]
            base = pts[0]
            tip = pts[-1]
            bone_lengths = [
                cls._safe_norm(pts[i + 1] - pts[i])
                for i in range(len(pts) - 1)
            ]
            chain_length = sum(bone_lengths)
            straight_distance = cls._safe_norm(tip - base)
            extension = straight_distance / max(chain_length, 1e-6)
            direction = cls._unit_vector(tip - base)

            features.extend(
                [
                    cls._safe_norm(tip - wrist),
                    straight_distance,
                    extension,
                    cls._joint_angle(pts[0], pts[1], pts[2]),
                    cls._joint_angle(pts[1], pts[2], pts[3]),
                    float(direction[0]),
                    float(direction[1]),
                    float(direction[2]),
                ]
            )
            directions.append(direction)
            tips.append(tip)

        for i in range(len(tips) - 1):
            features.append(cls._safe_norm(tips[i + 1] - tips[i]))

        for i in range(len(directions) - 1):
            features.append(float(np.clip(np.dot(directions[i], directions[i + 1]), -1.0, 1.0)))

        return np.asarray(features, dtype=np.float32)

    @staticmethod
    def _sample_points(points, count):
        if len(points) == 0:
            return np.zeros((count, 2), dtype=np.float32)

        center = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        order = np.argsort(angles)
        ordered = points[order]
        indices = np.linspace(0, len(ordered) - 1, count).astype(int)
        return ordered[indices].astype(np.float32)

    @staticmethod
    def _to_uint8_rgb(image_rgb):
        image = np.asarray(image_rgb)
        if image.dtype != np.uint8:
            image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
        return np.ascontiguousarray(image)

    def close(self):
        if self._hands is not None:
            self._hands.close()
        if self._task_landmarker is not None:
            self._task_landmarker.close()
