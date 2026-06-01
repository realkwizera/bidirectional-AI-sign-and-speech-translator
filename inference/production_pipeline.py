"""
PRODUCTION-GRADE ASL TRANSFORMER PIPELINE
Step 7+: Deployment with optimization, threading, and monitoring
"""

import torch
import cv2
import numpy as np
from collections import deque
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional
import threading
import time
import logging
from queue import Queue
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.asl_ctc_transformer import ASLCTCModel
from nlp.ctc_beam_search import CTCBeamSearch
from utils.config import DEVICE, ID_TO_CHAR, BEST_MODEL
from utils.hand_landmarks import HAND_LANDMARK_DIM, HandLandmarkExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Production pipeline configuration"""
    frame_size: int = 64
    buffer_size: int = 30
    beam_width: int = 5
    confidence_threshold: float = 0.5
    smoothing_window: int = 3
    prediction_stride: int = 15
    roi: Tuple[float, float, float, float] = (0.42, 0.05, 0.55, 0.90)
    min_hand_ratio: float = 0.012
    min_landmark_presence: float = 0.25
    min_margin: float = 0.04
    max_blank_ratio: float = 0.92
    use_landmarks: bool = True
    num_workers: int = 2
    use_fp16: bool = True
    num_frames_per_prediction: int = 30

class FrameBuffer:
    """Thread-safe frame buffer with smoothing"""
    def __init__(self, size: int, smoothing_window: int = 3):
        self.buffer = deque(maxlen=size)
        self.smoothing_window = smoothing_window
        self.predictions = deque(maxlen=smoothing_window)
        self.lock = threading.Lock()
    
    def add(self, frame: torch.Tensor):
        """Add frame to buffer (thread-safe)"""
        with self.lock:
            self.buffer.append(frame)
    
    def get(self):
        """Get stacked buffer or None if not full"""
        with self.lock:
            if len(self.buffer) < self.buffer.maxlen:
                return None
            return list(self.buffer)
    
    def is_full(self) -> bool:
        """Check if buffer is full"""
        with self.lock:
            return len(self.buffer) == self.buffer.maxlen
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()
    
    def add_prediction(self, text: str, confidence: float):
        """Add prediction for smoothing"""
        with self.lock:
            self.predictions.append((text, confidence))
    
    def get_smoothed_prediction(self) -> Optional[Tuple[str, float]]:
        """Get most common prediction (majority voting)"""
        with self.lock:
            if not self.predictions:
                return None
            
            texts = [p[0] for p in self.predictions]
            confidences = [p[1] for p in self.predictions]
            
            # Majority vote
            most_common = max(set(texts), key=texts.count)
            avg_conf = np.mean(confidences)
            
            return most_common, avg_conf

class ProductionPipeline:
    """Production-grade ASL recognition pipeline"""
    
    def __init__(self, config: PipelineConfig = None, model_path: str = str(BEST_MODEL)):
        self.config = config or PipelineConfig()
        self.device = DEVICE
        
        logger.info(f"Initializing production pipeline on {self.device}")
        
        # Load model
        self.landmark_extractor = (
            HandLandmarkExtractor(static_image_mode=False, allow_contour_fallback=False)
            if self.config.use_landmarks
            else None
        )
        self.model = ASLCTCModel(num_classes=28, use_landmarks=self.config.use_landmarks).to(self.device)
        self.model.eval()
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            missing, unexpected = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logger.info(f"✓ Model loaded from {model_path}")
            if missing or unexpected:
                logger.warning("Checkpoint does not fully match the hybrid landmark model; retrain is recommended.")
        else:
            logger.warning(f"⚠ Model not found at {model_path}")
        
        # Decoder
        self.beam_search = CTCBeamSearch(
            beam_width=self.config.beam_width,
            blank=0
        )
        
        # Buffers
        self.frame_buffer = FrameBuffer(
            size=self.config.buffer_size,
            smoothing_window=self.config.smoothing_window
        )
        
        # Queues for threading
        self.frame_queue = Queue(maxsize=10)
        self.prediction_queue = Queue()
        
        # Stats
        self.frame_count = 0
        self.prediction_count = 0
        self.inference_times = deque(maxlen=30)
        self.last_emitted = ""
        self.hand_presence = deque(maxlen=self.config.buffer_size)
        self.prev_roi_gray = None
        self.last_candidate = {
            "text": "",
            "confidence": 0.0,
            "class_text": "",
            "class_confidence": 0.0,
            "margin": 0.0,
            "blank_ratio": 1.0,
            "landmark_presence": 0.0,
            "reason": "warming up",
        }

    def _roi_bounds(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        h, w = frame.shape[:2]
        x, y, rw, rh = self.config.roi
        if max(self.config.roi) <= 1.0:
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + rw) * w)
            y2 = int((y + rh) * h)
        else:
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + rw)
            y2 = int(y + rh)

        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        return x1, y1, x2, y2

    def crop_hand_roi(self, frame: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = self._roi_bounds(frame)
        return frame[y1:y2, x1:x2]

    def estimate_hand_presence(self, frame: np.ndarray) -> float:
        roi = self.crop_hand_roi(frame)
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
        lower = np.array([10, 120, 55], dtype=np.uint8)
        upper = np.array([255, 190, 155], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower, upper)

        kernel = np.ones((3, 3), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_ratio = float(np.count_nonzero(skin_mask)) / float(skin_mask.size)

        mean, std = cv2.meanStdDev(gray)
        threshold = max(18.0, float(mean[0][0] + 0.45 * std[0][0]))
        _, bright_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_area = max((cv2.contourArea(contour) for contour in contours), default=0.0)
        bright_blob_ratio = largest_area / float(gray.size)

        motion_ratio = 0.0
        if self.prev_roi_gray is not None and self.prev_roi_gray.shape == gray.shape:
            diff = cv2.absdiff(gray, self.prev_roi_gray)
            _, motion_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_ratio = float(np.count_nonzero(motion_mask)) / float(motion_mask.size)
        self.prev_roi_gray = gray

        return max(skin_ratio, bright_blob_ratio, motion_ratio)

    def has_stable_hand(self) -> bool:
        if len(self.hand_presence) < min(10, self.hand_presence.maxlen):
            return False
        recent = list(self.hand_presence)[-10:]
        return sum(score >= self.config.min_hand_ratio for score in recent) >= 6

    def check_candidate(self, text: str, confidence: float, diagnostics: dict) -> Tuple[bool, str]:
        if not self.has_stable_hand():
            return False, "no stable hand"
        if diagnostics.get("landmark_presence", 0.0) < self.config.min_landmark_presence:
            return False, f"low landmarks {diagnostics.get('landmark_presence', 0.0):.2f}"
        if not text:
            class_text = diagnostics.get("class_text", "")
            class_confidence = diagnostics.get("class_confidence", 0.0)
            if class_text and class_confidence >= self.config.confidence_threshold:
                text = class_text
                confidence = class_confidence
            else:
                return False, "empty decode"
        if confidence < self.config.confidence_threshold:
            return False, f"low conf {confidence:.2f}"
        if diagnostics["margin"] < self.config.min_margin:
            return False, f"low margin {diagnostics['margin']:.2f}"
        if diagnostics["blank_ratio"] > self.config.max_blank_ratio:
            return False, f"blank {diagnostics['blank_ratio']:.2f}"
        if text == self.last_emitted:
            return False, "duplicate"
        return True, "emit"
    
    def preprocess_frame(self, frame: np.ndarray):
        """Preprocess frame: resize, normalize, channel-first"""
        frame = self.crop_hand_roi(frame)

        # Resize
        frame = cv2.resize(frame, (self.config.frame_size, self.config.frame_size))
        
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = self.extract_landmarks(frame)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        # Channel first
        frame = np.transpose(frame, (2, 0, 1))
        
        return torch.tensor(frame, dtype=torch.float32), landmarks

    def extract_landmarks(self, frame_rgb: np.ndarray) -> torch.Tensor:
        if not self.config.use_landmarks or self.landmark_extractor is None:
            return torch.zeros(HAND_LANDMARK_DIM, dtype=torch.float32)
        result = self.landmark_extractor.extract(frame_rgb)
        return torch.tensor(result.features, dtype=torch.float32)
    
    @torch.no_grad()
    def predict_batch(self, frames: torch.Tensor, landmarks: torch.Tensor) -> Tuple[str, float, dict]:
        """Predict from frame batch (B, T, C, H, W)"""
        if frames.shape[1] != self.config.num_frames_per_prediction:
            raise ValueError(f"Expected {self.config.num_frames_per_prediction} frames")
        
        frames = frames.unsqueeze(0).to(self.device)  # Add batch dim
        landmarks = landmarks.unsqueeze(0).to(self.device)
        
        start_time = time.time()
        
        # Forward pass
        with torch.amp.autocast(
            device_type='cuda',
            enabled=self.config.use_fp16 and self.device.type == 'cuda',
        ):
            outputs = self.model(frames, landmarks if self.config.use_landmarks else None, return_aux=True)
            logits = outputs["ctc_logits"]
            class_probs = torch.softmax(outputs["class_logits"], dim=-1).squeeze(0)
            class_id = int(class_probs.argmax().item())
            class_text = ID_TO_CHAR.get(class_id, "")
            class_confidence = float(class_probs[class_id].item())
        
        # Decode
        probs = torch.softmax(logits, dim=-1)
        probs = probs.squeeze(0).cpu().numpy()
        
        text = self.beam_search.decode(probs, ID_TO_CHAR)
        
        non_blank = probs[:, 1:]
        non_blank_max = np.max(non_blank, axis=1)
        blank_probs = probs[:, 0]
        top2 = np.sort(np.partition(probs, -2, axis=1)[:, -2:], axis=1)
        margin = float(np.mean(top2[:, 1] - top2[:, 0]))
        confidence = float(np.mean(non_blank_max))
        blank_ratio = float(np.mean(blank_probs > non_blank_max))
        landmark_presence = float(landmarks[:, :, -1].mean()) if landmarks.numel() else 0.0
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return text, confidence, {
            "margin": margin,
            "blank_ratio": blank_ratio,
            "landmark_presence": landmark_presence,
            "class_text": class_text,
            "class_confidence": class_confidence,
        }
    
    def process_frame(self, frame: np.ndarray) -> Optional[Tuple[str, float]]:
        """Process single frame, return prediction if buffer full"""
        self.frame_count += 1
        hand_score = self.estimate_hand_presence(frame)
        self.hand_presence.append(hand_score)
        
        # Preprocess
        tensor, landmarks = self.preprocess_frame(frame)
        self.frame_buffer.add((tensor, landmarks))
        
        # Check if buffer full
        if not self.frame_buffer.is_full():
            return None

        if self.frame_count % self.config.prediction_stride != 0:
            return None
        
        # Predict
        batch = self.frame_buffer.get()
        if batch is not None:
            frames = torch.stack([item[0] for item in batch])
            landmarks = torch.stack([item[1] for item in batch])
            text, confidence, diagnostics = self.predict_batch(frames, landmarks)
            self.prediction_count += 1
            is_reliable, reason = self.check_candidate(text, confidence, diagnostics)
            self.last_candidate = {
                "text": text or "",
                "confidence": confidence,
                "class_text": diagnostics["class_text"],
                "class_confidence": diagnostics["class_confidence"],
                "margin": diagnostics["margin"],
                "blank_ratio": diagnostics["blank_ratio"],
                "landmark_presence": diagnostics["landmark_presence"],
                "reason": reason,
            }
            
            # Add to smoothing buffer
            emitted_text = text or diagnostics["class_text"]
            emitted_confidence = confidence if text else diagnostics["class_confidence"]
            self.frame_buffer.add_prediction(emitted_text, emitted_confidence)
            
            # Get smoothed prediction
            smoothed = self.frame_buffer.get_smoothed_prediction()
            if smoothed:
                text, confidence = smoothed
                is_reliable, reason = self.check_candidate(text, confidence, diagnostics)
                self.last_candidate["reason"] = reason
                if is_reliable:
                    self.last_emitted = text
                    return text, confidence
                return None
            
            if is_reliable:
                self.last_emitted = emitted_text
                return emitted_text, emitted_confidence
        
        return None
    
    def get_fps(self) -> float:
        """Get average inference FPS"""
        if not self.inference_times:
            return 0.0
        avg_time = np.mean(self.inference_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def run_webcam(self, window_name: str = "ASL Transformer"):
        """Run real-time webcam inference"""
        logger.info("Starting webcam inference...")
        logger.info("Press 'q' to quit, 'r' to reset buffer, 's' to save frame")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        predictions_history = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Frame read failed")
                break
            
            # Process
            result = self.process_frame(frame)
            
            # Display
            display_frame = frame.copy()
            
            # Buffer status
            x1, y1, x2, y2 = self._roi_bounds(display_frame)
            roi_color = (0, 255, 0) if self.has_stable_hand() else (0, 165, 255)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), roi_color, 2)

            cv2.putText(
                display_frame,
                f"Buffer: {len(self.frame_buffer.buffer)}/{self.config.buffer_size}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # FPS
            fps = self.get_fps()
            cv2.putText(
                display_frame,
                f"FPS: {fps:.1f} Hand: {self.hand_presence[-1] if self.hand_presence else 0.0:.3f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            candidate_text = self.last_candidate["text"] or "--"
            cv2.putText(
                display_frame,
                f"Cand: {candidate_text}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Cls: {self.last_candidate['class_text'] or '--'} "
                f"{self.last_candidate['class_confidence']:.2f}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                (
                    f"c:{self.last_candidate['confidence']:.2f} "
                    f"m:{self.last_candidate['margin']:.2f} "
                    f"b:{self.last_candidate['blank_ratio']:.2f} "
                    f"lm:{self.last_candidate['landmark_presence']:.2f}"
                ),
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                display_frame,
                f"Gate: {self.last_candidate['reason']}",
                (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )
            
            # Prediction
            if result:
                text, confidence = result
                predictions_history.append(text)
                
                color = (0, 255, 0) if confidence >= self.config.confidence_threshold else (0, 165, 255)
                
                cv2.putText(
                    display_frame,
                    f"Text: {text} (conf: {confidence:.2f})",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )
                
                logger.info(f"[{self.frame_count:05d}] {text:30s} (conf: {confidence:.2f})")
            
            cv2.imshow(window_name, display_frame)
            
            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.frame_buffer.clear()
                self.last_emitted = ""
                self.hand_presence.clear()
                self.prev_roi_gray = None
                self.last_candidate["reason"] = "reset"
                logger.info("Buffer reset")
            elif key == ord('s'):
                cv2.imwrite(f"frame_{self.frame_count}.png", frame)
                logger.info(f"Frame saved: frame_{self.frame_count}.png")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Summary
        print("\n" + "="*70)
        print("SESSION SUMMARY")
        print("="*70)
        print(f"Total frames: {self.frame_count}")
        print(f"Predictions: {self.prediction_count}")
        print(f"Average FPS: {self.get_fps():.1f}")
        if predictions_history:
            print(f"All predictions: {' → '.join(predictions_history)}")
        print("="*70)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production ASL Pipeline")
    parser.add_argument('--model', type=str, default=str(BEST_MODEL),
                       help='Path to model checkpoint')
    parser.add_argument('--beam-width', type=int, default=5,
                       help='Beam width for decoding')
    parser.add_argument('--frame-size', type=int, default=64,
                       help='Frame size (64x64)')
    parser.add_argument('--fp16', action='store_true',
                       help='Use FP16 mixed precision')
    parser.add_argument('--stride', type=int, default=15,
                       help='Frames between predictions once the buffer is full')
    parser.add_argument('--roi', type=str, default='0.42,0.05,0.55,0.90',
                       help='Hand ROI as x,y,w,h. Use ratios 0-1 or pixel values.')
    parser.add_argument('--min-hand-ratio', type=float, default=0.012,
                       help='Minimum skin/hand-like pixel ratio inside ROI')
    parser.add_argument('--min-landmark-presence', type=float, default=0.25,
                       help='Minimum real hand landmark confidence before emitting text')
    parser.add_argument('--min-margin', type=float, default=0.04,
                       help='Minimum average top-1/top-2 probability margin')
    parser.add_argument('--no-landmarks', action='store_true',
                       help='Disable MediaPipe hand landmarks and use CNN-only inference')
    
    args = parser.parse_args()
    
    config = PipelineConfig(
        beam_width=args.beam_width,
        frame_size=args.frame_size,
        use_fp16=args.fp16,
        prediction_stride=args.stride,
        roi=tuple(float(part.strip()) for part in args.roi.split(',')),
        min_hand_ratio=args.min_hand_ratio,
        min_landmark_presence=args.min_landmark_presence,
        min_margin=args.min_margin,
        use_landmarks=not args.no_landmarks,
    )
    
    try:
        pipeline = ProductionPipeline(config=config, model_path=args.model)
    except RuntimeError as exc:
        print(str(exc))
        raise SystemExit(1)
    pipeline.run_webcam()
