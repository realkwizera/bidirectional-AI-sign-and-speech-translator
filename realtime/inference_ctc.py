"""
Native OpenCV ASL app:
1. Sign-to-speech from live webcam letters.
2. Speech/text-to-sign using images from data/images.
"""

from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue
import random
import sys
import threading
import time

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_ROOT = PROJECT_ROOT / "data" / "images"

from models.asl_ctc_transformer import ASLCTCModel
from nlp.ctc_beam_search import CTCBeamSearch
from utils.config import BEST_MODEL, DEVICE, ID_TO_CHAR
from utils.hand_landmarks import HAND_LANDMARK_DIM, HAND_PRESENCE_INDEX, HandLandmarkExtractor


LETTER_IDS = {idx for idx, char in ID_TO_CHAR.items() if char and char != " "}
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


class FingerspellingInference:
    def __init__(
        self,
        model_path=BEST_MODEL,
        beam_width=5,
        confidence_threshold=0.55,
        stride=15,
        roi=(0.0, 0.0, 1.0, 1.0),
        min_hand_ratio=0.012,
        min_landmark_presence=0.25,
        min_margin=0.04,
        max_blank_ratio=0.92,
        use_landmarks=True,
        letter_interval=1.0,
        speak_on_no_hand=True,
        mirror=True,
        mirror_prediction=False,
        no_hand_frames=8,
        crop_padding=0.35,
        mic_index=None,
        mic_timeout=10.0,
        mic_phrase_time_limit=12.0,
        mic_ambient_duration=0.2,
        mic_energy_threshold=400.0,
        mic_dynamic_energy=False,
    ):
        self.device = DEVICE
        self.use_landmarks = use_landmarks
        self.landmark_extractor = (
            HandLandmarkExtractor(static_image_mode=False, allow_contour_fallback=False)
            if use_landmarks
            else None
        )
        self.landmark_backend = getattr(self.landmark_extractor, "backend", "off") if self.landmark_extractor else "off"
        self.model = ASLCTCModel(num_classes=28, use_landmarks=use_landmarks).to(self.device)
        self.model.eval()

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            try:
                missing, unexpected = self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            except RuntimeError as exc:
                if "landmark_encoder.0.weight" in str(exc):
                    print("Checkpoint uses the old landmark size. Retrain with: python train_ctc.py --epochs 30")
                    raise SystemExit(1) from exc
                raise
            print(f"Model loaded from {model_path}")
            print(f"Landmarks: {'ON' if self.use_landmarks else 'OFF'} ({self.landmark_backend})")
            config = checkpoint.get("config", {})
            if config.get("preprocessing_roi") not in (None, "mediapipe_tight_hand_crop"):
                print("WARNING: checkpoint was not trained with the current tight hand crop.")
            if missing or unexpected:
                print("Checkpoint partially matched the current model; retraining is recommended.")
        else:
            print(f"Model not found at {model_path}")
            print("  Train a model first: python train_ctc.py")

        self.beam_search = CTCBeamSearch(beam_width=beam_width, blank=0)
        self.confidence_threshold = confidence_threshold
        self.stride = stride
        self.roi = roi
        self.min_hand_ratio = min_hand_ratio
        self.min_landmark_presence = min_landmark_presence
        self.min_margin = min_margin
        self.max_blank_ratio = max_blank_ratio
        self.letter_interval = letter_interval
        self.speak_on_no_hand = speak_on_no_hand
        self.mirror_display = mirror
        self.mirror_prediction = mirror_prediction
        self.no_hand_frames_required = no_hand_frames
        self.crop_padding = crop_padding
        self.mic_index = mic_index
        self.mic_timeout = mic_timeout
        self.mic_phrase_time_limit = mic_phrase_time_limit
        self.mic_ambient_duration = mic_ambient_duration
        self.mic_energy_threshold = mic_energy_threshold
        self.mic_dynamic_energy = mic_dynamic_energy

        self.buffer = deque(maxlen=30)
        self.landmark_buffer = deque(maxlen=30)
        self.hand_presence = deque(maxlen=30)
        self.frame_count = 0
        self.predictions = []
        self.current_letters = []
        self.completed_sets = []
        self.second_votes = []
        self.second_started_at = time.monotonic()
        self.hand_was_present = False
        self.no_hand_frames = 0
        self.last_emitted = ""
        self.state_lock = threading.Lock()
        self.speech_queue = Queue()
        self.speech_busy = False
        self.app_mode = "sign_to_speech"
        self.sts_text = ""
        self.sts_status = "Press M to record microphone"
        self.sts_recording = False
        self.sts_started_at = time.monotonic()
        self.sts_letter_index = 0
        self.sign_image_cache = {}
        self.last_candidate = self.empty_candidate("warming up")

    @staticmethod
    def empty_candidate(reason):
        return {
            "text": "",
            "confidence": 0.0,
            "class_text": "",
            "class_confidence": 0.0,
            "margin": 0.0,
            "blank_ratio": 1.0,
            "landmark_presence": 0.0,
            "top": "",
            "reason": reason,
        }

    @staticmethod
    def put_latest(queue, item):
        try:
            queue.put_nowait(item)
        except Full:
            try:
                queue.get_nowait()
                queue.task_done()
            except Empty:
                pass
            queue.put_nowait(item)

    def speak_text(self, text):
        if self.speak_on_no_hand and text.strip():
            self.speech_queue.put(text)

    def speech_worker(self, stop_event):
        engine = None
        while True:
            text = self.speech_queue.get()
            if text is None:
                self.speech_queue.task_done()
                break
            if text.strip():
                self.speech_busy = True
                try:
                    if engine is None:
                        import pyttsx3
                        engine = pyttsx3.init()
                    engine.say(text)
                    engine.runAndWait()
                except Exception as exc:
                    print(f"Speech failed: {exc}")
                finally:
                    self.speech_busy = False
            self.speech_queue.task_done()

    def reset_second_votes(self):
        self.second_votes.clear()
        self.second_started_at = time.monotonic()

    def choose_second_letter(self):
        if not self.second_votes:
            return "", 0.0
        counts = {}
        for letter, confidence in self.second_votes:
            counts.setdefault(letter, {"count": 0, "best_conf": 0.0})
            counts[letter]["count"] += 1
            counts[letter]["best_conf"] = max(counts[letter]["best_conf"], confidence)
        letter, stats = max(counts.items(), key=lambda item: (item[1]["count"], item[1]["best_conf"]))
        return letter, stats["best_conf"]

    def update_letter_set(self, diagnostics):
        letter = diagnostics.get("class_text", "")
        confidence = diagnostics.get("class_confidence", 0.0)
        if letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" and confidence >= self.confidence_threshold:
            self.second_votes.append((letter, confidence))
        if time.monotonic() - self.second_started_at < self.letter_interval:
            return
        chosen, chosen_conf = self.choose_second_letter()
        self.reset_second_votes()
        if not chosen:
            return
        self.current_letters.append(chosen)
        self.predictions.append(chosen)
        self.last_emitted = chosen
        print(f"[Second {len(self.current_letters):03d}] {chosen} (conf: {chosen_conf:.2f}, word: {''.join(self.current_letters)})")

    def flush_current_letters(self):
        if not self.current_letters:
            return
        text = "".join(self.current_letters)
        self.completed_sets.append(text)
        print(f"Hand lost. Completed letters: {text}")
        self.speak_text(text)
        self.current_letters = []
        self.reset_second_votes()
        self.last_emitted = ""

    def _roi_bounds(self, frame):
        h, w = frame.shape[:2]
        x, y, rw, rh = self.roi
        if max(self.roi) <= 1.0:
            x1, y1, x2, y2 = int(x * w), int(y * h), int((x + rw) * w), int((y + rh) * h)
        else:
            x1, y1, x2, y2 = int(x), int(y), int(x + rw), int(y + rh)
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))
        return x1, y1, x2, y2

    def crop_hand_roi(self, frame):
        x1, y1, x2, y2 = self._roi_bounds(frame)
        return frame[y1:y2, x1:x2]

    def _display_roi_bounds(self, frame):
        x1, y1, x2, y2 = self._roi_bounds(frame)
        if self.mirror_display != self.mirror_prediction:
            width = frame.shape[1]
            return width - x2, y1, width - x1, y2
        return x1, y1, x2, y2

    def preprocess_frame(self, frame):
        search_frame = self.crop_hand_roi(frame)
        frame_rgb = cv2.cvtColor(search_frame, cv2.COLOR_BGR2RGB)
        landmark_points = None
        landmarks = torch.zeros(HAND_LANDMARK_DIM, dtype=torch.float32)
        if self.use_landmarks and self.landmark_extractor is not None:
            result = self.landmark_extractor.extract(frame_rgb)
            landmark_points = result.landmarks
            if result.landmarks is not None and result.presence >= self.min_landmark_presence:
                frame_rgb, crop_points, _ = HandLandmarkExtractor.crop_to_landmarks(
                    frame_rgb,
                    result.landmarks,
                    padding=self.crop_padding,
                )
                features = HandLandmarkExtractor.features_from_landmarks(crop_points, result.presence)
                landmarks = torch.tensor(features, dtype=torch.float32)
        frame = cv2.resize(frame_rgb, (64, 64)).astype(np.float32) / 255.0
        frame = np.transpose(frame, (2, 0, 1))
        return torch.tensor(frame, dtype=torch.float32), landmarks, landmark_points

    @torch.no_grad()
    def predict_sequence(self):
        if len(self.buffer) < 30:
            return None, None, self.empty_candidate("warming up")
        seq = torch.stack(list(self.buffer)).unsqueeze(0).to(self.device)
        landmarks = torch.stack(list(self.landmark_buffer)).unsqueeze(0).to(self.device)
        outputs = self.model(seq, landmarks if self.use_landmarks else None, return_aux=True)
        logits = outputs["ctc_logits"]
        class_probs = torch.softmax(outputs["class_logits"], dim=-1).squeeze(0)
        if LETTER_IDS:
            mask = torch.ones_like(class_probs, dtype=torch.bool)
            mask[list(LETTER_IDS)] = False
            class_probs = class_probs.masked_fill(mask, 0.0)
        class_id = int(class_probs.argmax().item())
        class_text = ID_TO_CHAR.get(class_id, "")
        class_confidence = float(class_probs[class_id].item())
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        probs_np = probs.cpu().numpy()
        text = self.beam_search.decode(probs_np, ID_TO_CHAR)
        non_blank = probs[:, 1:]
        non_blank_max = non_blank.max(dim=1)[0]
        blank_probs = probs[:, 0]
        top2 = torch.topk(probs, k=2, dim=1).values
        mean_probs = probs.mean(dim=0)
        top_indices = torch.argsort(mean_probs, descending=True)[:3].tolist()
        diagnostics = {
            "margin": float((top2[:, 0] - top2[:, 1]).mean().item()),
            "blank_ratio": float((blank_probs > non_blank_max).float().mean().item()),
            "landmark_presence": float(landmarks[:, :, HAND_PRESENCE_INDEX].mean().item()) if landmarks.numel() else 0.0,
            "top": " ".join(f"{ID_TO_CHAR.get(int(idx), '<blank>') or '<blank>'}:{mean_probs[int(idx)].item():.2f}" for idx in top_indices),
            "class_text": class_text,
            "class_confidence": class_confidence,
        }
        return text, float(non_blank_max.mean().item()), diagnostics

    def has_stable_hand(self):
        if len(self.hand_presence) < min(10, self.hand_presence.maxlen):
            return False
        return sum(score >= self.min_hand_ratio for score in list(self.hand_presence)[-10:]) >= 6

    def check_candidate(self, text, confidence, diagnostics):
        if not self.has_stable_hand():
            return False, "no stable hand"
        if diagnostics.get("landmark_presence", 0.0) < self.min_landmark_presence:
            return False, f"low landmarks {diagnostics.get('landmark_presence', 0.0):.2f}"
        class_text = diagnostics.get("class_text", "")
        class_confidence = diagnostics.get("class_confidence", 0.0)
        if class_text and class_confidence >= self.confidence_threshold:
            return True, "emit"
        if not text:
            return False, "empty letter"
        if confidence < self.confidence_threshold:
            return False, f"low conf {confidence:.2f}"
        if diagnostics["margin"] < self.min_margin:
            return False, f"low margin {diagnostics['margin']:.2f}"
        if diagnostics["blank_ratio"] > self.max_blank_ratio:
            return False, f"blank {diagnostics['blank_ratio']:.2f}"
        if text == self.last_emitted:
            return False, "duplicate"
        return True, "emit"

    def reset_live_state(self):
        self.buffer.clear()
        self.landmark_buffer.clear()
        self.predictions.clear()
        self.current_letters.clear()
        self.completed_sets.clear()
        self.reset_second_votes()
        self.last_emitted = ""
        self.hand_presence.clear()
        self.hand_was_present = False
        self.no_hand_frames = 0
        self.last_candidate = self.empty_candidate("reset")

    def process_live_frame(self, frame):
        prediction_frame = cv2.flip(frame, 1) if self.mirror_prediction else frame
        display_frame = cv2.flip(frame, 1) if self.mirror_display else frame
        self.frame_count += 1
        tensor, landmarks, landmark_points = self.preprocess_frame(prediction_frame)
        landmark_presence = float(landmarks[HAND_PRESENCE_INDEX].item()) if landmarks.numel() else 0.0
        hand_present_now = landmark_presence >= self.min_landmark_presence
        self.hand_presence.append(1.0 if hand_present_now else 0.0)
        display_frame = self.draw_hand_landmarks(display_frame, landmark_points)
        if not hand_present_now:
            self.no_hand_frames += 1
            self.buffer.clear()
            self.landmark_buffer.clear()
            self.reset_second_votes()
            self.last_candidate["reason"] = f"no hand {self.no_hand_frames}/{self.no_hand_frames_required}"
            if self.no_hand_frames >= self.no_hand_frames_required:
                if self.hand_was_present:
                    self.flush_current_letters()
                self.hand_was_present = False
            return display_frame, self.state(False, landmark_presence)

        self.no_hand_frames = 0
        self.hand_was_present = True
        self.buffer.append(tensor)
        self.landmark_buffer.append(landmarks)
        if len(self.buffer) == 30:
            text, conf, diagnostics = self.predict_sequence()
            is_reliable, reason = self.check_candidate(text, conf, diagnostics)
            self.last_candidate = {
                "text": text or "",
                "confidence": conf or 0.0,
                "class_text": diagnostics["class_text"],
                "class_confidence": diagnostics["class_confidence"],
                "margin": diagnostics["margin"],
                "blank_ratio": diagnostics["blank_ratio"],
                "landmark_presence": diagnostics["landmark_presence"],
                "top": diagnostics["top"],
                "reason": reason,
            }
            if is_reliable:
                self.update_letter_set(diagnostics)
        return display_frame, self.state(self.has_stable_hand(), landmark_presence)

    def state(self, has_stable_hand, hand_score):
        return {
            "buffer_len": len(self.buffer),
            "hand_score": hand_score,
            "has_stable_hand": has_stable_hand,
            "last_candidate": dict(self.last_candidate),
            "last_prediction": self.predictions[-1] if self.predictions else "",
            "all_predictions": list(self.predictions),
            "current_letters": "".join(self.current_letters),
            "completed_sets": list(self.completed_sets),
            "frame_count": self.frame_count,
        }

    @staticmethod
    def fit_text(text, max_chars):
        return text if len(text) <= max_chars else "..." + text[-max(0, max_chars - 3):]

    @staticmethod
    def sign_letters_from_text(text):
        return [char for char in text.upper() if "A" <= char <= "Z"]

    def sign_image_for_letter(self, letter):
        if letter in self.sign_image_cache:
            return self.sign_image_cache[letter]
        folder = IMAGE_ROOT / letter
        if not folder.exists():
            matches = [path for path in IMAGE_ROOT.iterdir() if path.is_dir() and path.name.upper() == letter]
            folder = matches[0] if matches else folder
        if not folder.exists():
            self.sign_image_cache[letter] = None
            return None
        images = [path for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
        self.sign_image_cache[letter] = random.choice(images) if images else None
        return self.sign_image_cache[letter]

    def start_speech_to_sign_recording(self):
        if self.sts_recording:
            return
        try:
            import pyaudio  # noqa: F401
        except ImportError:
            self.sts_status = "PyAudio not installed; microphone input is unavailable"
            return
        self.sts_recording = True
        mic_label = f"mic {self.mic_index}" if self.mic_index is not None else "default mic"
        self.sts_status = f"Listening on {mic_label}..."
        threading.Thread(target=self.speech_to_sign_worker, daemon=True).start()

    def speech_to_sign_worker(self):
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.dynamic_energy_threshold = self.mic_dynamic_energy
            recognizer.pause_threshold = 0.8
            recognizer.phrase_threshold = 0.35
            recognizer.non_speaking_duration = 0.4
            if self.mic_energy_threshold is not None:
                recognizer.energy_threshold = self.mic_energy_threshold
            with sr.Microphone(device_index=self.mic_index) as source:
                if self.mic_dynamic_energy and self.mic_ambient_duration > 0:
                    recognizer.adjust_for_ambient_noise(source, duration=self.mic_ambient_duration)
                self.sts_status = f"Listening now... threshold {int(recognizer.energy_threshold)}"
                audio = recognizer.listen(source, timeout=self.mic_timeout, phrase_time_limit=self.mic_phrase_time_limit)
            self.sts_status = "Recognizing..."
            self.sts_text = recognizer.recognize_google(audio)
            self.sts_letter_index = 0
            self.sts_started_at = time.monotonic()
            self.sts_status = "Speech captured"
        except sr.WaitTimeoutError:
            self.sts_status = "No voice detected; try --mic-index or speak closer"
        except sr.UnknownValueError:
            self.sts_status = "Voice heard, but speech was not understood"
        except Exception as exc:
            self.sts_status = f"Microphone failed: {exc}"
        finally:
            self.sts_recording = False

    @staticmethod
    def put_panel_text(canvas, text, y, scale=0.65, color=(235, 235, 235), thickness=2):
        cv2.putText(canvas, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    @staticmethod
    def letterbox(image, width, height):
        h, w = image.shape[:2]
        scale = min(width / max(w, 1), height / max(h, 1))
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h))
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        x, y = (width - new_w) // 2, (height - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas

    def draw_overlay(self, frame, state):
        canvas = np.zeros((720, 1000, 3), dtype=np.uint8)
        canvas[:500, :] = self.letterbox(frame, 1000, 500)
        canvas[500:, :] = (22, 22, 22)
        candidate = state["last_candidate"]
        all_predictions = " -> ".join(state["all_predictions"]) or "--"
        completed_sets = " | ".join(state["completed_sets"]) or "--"
        hand_text = "hand detected" if state["has_stable_hand"] else "no hand"
        self.put_panel_text(canvas, "Sign to Speech", 535, 0.75, (100, 220, 120))
        self.put_panel_text(canvas, f"Current: {state['current_letters'] or '--'}", 575)
        self.put_panel_text(canvas, f"All predictions: {self.fit_text(all_predictions, 70)}", 610)
        self.put_panel_text(canvas, f"Completed sets: {self.fit_text(completed_sets, 70)}", 645)
        self.put_panel_text(
            canvas,
            f"Last: {state['last_prediction'] or '--'}   Status: {hand_text}   Candidate: {candidate['class_text'] or '--'} {candidate['class_confidence']:.2f}",
            680,
            0.55,
            (190, 190, 190),
        )
        self.put_panel_text(canvas, "1 Sign-to-Speech | 2 Speech-to-Sign | M Mic | R Reset | Q Quit", 710, 0.52, (160, 160, 160), 1)
        return canvas

    def draw_speech_to_sign(self):
        canvas = np.zeros((720, 1000, 3), dtype=np.uint8)
        canvas[:220, :] = (22, 22, 22)
        canvas[220:, :] = (8, 8, 8)
        letters = self.sign_letters_from_text(self.sts_text)
        if letters and time.monotonic() - self.sts_started_at >= 0.8:
            self.sts_letter_index = (self.sts_letter_index + 1) % len(letters)
            self.sts_started_at = time.monotonic()
        self.put_panel_text(canvas, "Speech/Text to Sign", 42, 0.8, (100, 180, 255))
        self.put_panel_text(canvas, "M Record microphone | C Clear | 1 Sign-to-Speech | Q Quit", 82, 0.55, (180, 180, 180), 1)
        self.put_panel_text(canvas, f"Mic: {self.fit_text(self.sts_status, 84)}", 122, 0.58)
        self.put_panel_text(canvas, f"Text: {self.fit_text(self.sts_text or '--', 92)}", 160, 0.62)
        self.put_panel_text(canvas, f"Letters: {' '.join(letters) if letters else '--'}", 198, 0.58, (220, 220, 220))
        if not letters:
            self.put_panel_text(canvas, "Press M and speak, then signs will play here.", 470, 0.75, (180, 180, 180))
            return canvas
        letter = letters[self.sts_letter_index]
        image_path = self.sign_image_for_letter(letter)
        if image_path is None:
            self.put_panel_text(canvas, f"No sign image found for {letter}", 470, 0.8, (80, 80, 255))
            return canvas
        image = cv2.imread(str(image_path))
        if image is None:
            self.put_panel_text(canvas, f"Could not read image for {letter}", 470, 0.8, (80, 80, 255))
            return canvas
        canvas[250:680, 140:860] = self.letterbox(image, 720, 430)
        cv2.putText(canvas, letter, (470, 705), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        return canvas

    def draw_hand_landmarks(self, frame, landmark_points):
        if landmark_points is None or len(landmark_points) < 21:
            return frame
        x1, y1, x2, y2 = self._display_roi_bounds(frame)
        roi_w, roi_h = max(1, x2 - x1), max(1, y2 - y1)
        points = []
        for point in landmark_points[:21]:
            point_x = 1.0 - float(point[0]) if self.mirror_display != self.mirror_prediction else float(point[0])
            points.append((int(x1 + point_x * roi_w), int(y1 + float(point[1]) * roi_h)))
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 255, 255), 2)
        for idx, point in enumerate(points):
            color = (0, 255, 0) if idx in {4, 8, 12, 16, 20} else (255, 255, 255)
            cv2.circle(frame, point, 4, color, -1)
        return frame

    def camera_worker(self, cap, frame_queue, stop_event):
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                self.put_latest(frame_queue, None)
                stop_event.set()
                break
            self.put_latest(frame_queue, frame)

    def inference_worker(self, frame_queue, prediction_queue, stop_event):
        while not stop_event.is_set() or not frame_queue.empty():
            try:
                frame = frame_queue.get(timeout=0.1)
            except Empty:
                continue
            try:
                if frame is None:
                    stop_event.set()
                    break
                with self.state_lock:
                    display_frame, state = self.process_live_frame(frame)
                self.put_latest(prediction_queue, (display_frame, state))
            except Exception as exc:
                print(f"Inference worker error: {exc}")
                stop_event.set()
                break
            finally:
                frame_queue.task_done()

    def run_webcam(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open webcam")
            return
        print("Webcam opened")
        print("  - 1: Sign-to-Speech")
        print("  - 2: Speech-to-Sign")
        print("  - m: record microphone in Speech-to-Sign")
        print("  - c: clear Speech-to-Sign text")
        print("  - q: quit, r: reset sign-to-speech")
        frame_queue, prediction_queue = Queue(maxsize=2), Queue(maxsize=2)
        stop_event = threading.Event()
        camera_thread = threading.Thread(target=self.camera_worker, args=(cap, frame_queue, stop_event), daemon=True)
        inference_thread = threading.Thread(target=self.inference_worker, args=(frame_queue, prediction_queue, stop_event), daemon=True)
        speech_thread = threading.Thread(target=self.speech_worker, args=(stop_event,), daemon=True)
        camera_thread.start()
        inference_thread.start()
        speech_thread.start()
        latest_frame = latest_state = latest_screen = None
        try:
            while not stop_event.is_set():
                try:
                    latest_frame, latest_state = prediction_queue.get(timeout=0.03)
                    prediction_queue.task_done()
                except Empty:
                    pass
                if self.app_mode == "sign_to_speech":
                    if latest_frame is not None and latest_state is not None:
                        latest_screen = self.draw_overlay(latest_frame, latest_state)
                        screen = latest_screen
                    elif latest_screen is not None:
                        screen = latest_screen
                    else:
                        screen = np.zeros((720, 1000, 3), dtype=np.uint8)
                        self.put_panel_text(screen, "Starting camera...", 360, 0.8)
                else:
                    screen = self.draw_speech_to_sign()
                cv2.imshow("ASL Translator", screen)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break
                if key == ord("1"):
                    self.app_mode = "sign_to_speech"
                if key == ord("2"):
                    self.app_mode = "speech_to_sign"
                if key == ord("m"):
                    self.app_mode = "speech_to_sign"
                    self.start_speech_to_sign_recording()
                if key == ord("c"):
                    self.sts_text = ""
                    self.sts_status = "Cleared"
                    self.sts_letter_index = 0
                if key == ord("r"):
                    with self.state_lock:
                        self.reset_live_state()
                    print("Buffer reset")
        finally:
            stop_event.set()
            camera_thread.join(timeout=1.0)
            inference_thread.join(timeout=1.0)
            with self.state_lock:
                self.flush_current_letters()
            self.speech_queue.put(None)
            self.speech_queue.join()
            speech_thread.join(timeout=1.0)
            cap.release()
            cv2.destroyAllWindows()
        print("\n" + "=" * 50)
        print("Recognition Summary")
        print("=" * 50)
        print(f"Total predictions: {len(self.predictions)}")
        print(f"All predictions: {' -> '.join(self.predictions)}")
        print(f"Completed sets: {' | '.join(self.completed_sets)}")


def print_microphones():
    try:
        import speech_recognition as sr
        for idx, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"{idx:02d}: {name}")
    except Exception as exc:
        print(f"Could not list microphones: {exc}")


def test_microphone_levels(device_index, seconds):
    import audioop
    import pyaudio

    audio = pyaudio.PyAudio()
    stream = None
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=1024,
        )
        print("Mic level test. Stay quiet for 2 seconds, then speak.")
        end_at = time.monotonic() + seconds
        while time.monotonic() < end_at:
            data = stream.read(1024, exception_on_overflow=False)
            print(f"rms={audioop.rms(data, 2)}")
            time.sleep(0.2)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fingerspelling recognition")
    parser.add_argument("--mode", default="webcam", choices=["webcam"])
    parser.add_argument("--model", type=str, default=str(BEST_MODEL))
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--letter-interval", type=float, default=1.0)
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--mirror-prediction", action="store_true")
    parser.add_argument("--roi", type=str, default="0.0,0.0,1.0,1.0")
    parser.add_argument("--min-hand-ratio", type=float, default=0.012)
    parser.add_argument("--min-landmark-presence", type=float, default=0.25)
    parser.add_argument("--no-hand-frames", type=int, default=8)
    parser.add_argument("--crop-padding", type=float, default=0.35)
    parser.add_argument("--min-margin", type=float, default=0.04)
    parser.add_argument("--no-landmarks", action="store_true")
    parser.add_argument("--list-mics", action="store_true")
    parser.add_argument("--mic-index", type=int, default=None)
    parser.add_argument("--mic-timeout", type=float, default=10.0)
    parser.add_argument("--mic-phrase-time-limit", type=float, default=12.0)
    parser.add_argument("--mic-ambient-duration", type=float, default=0.2)
    parser.add_argument("--mic-energy-threshold", type=float, default=None)
    parser.add_argument("--mic-dynamic-energy", action="store_true")
    parser.add_argument("--mic-test", action="store_true")
    parser.add_argument("--mic-test-seconds", type=float, default=6.0)
    args = parser.parse_args()

    if args.list_mics:
        print_microphones()
        raise SystemExit(0)
    if args.mic_test:
        test_microphone_levels(args.mic_index, args.mic_test_seconds)
        raise SystemExit(0)

    inference = FingerspellingInference(
        model_path=args.model,
        beam_width=args.beam_width,
        confidence_threshold=args.confidence_threshold,
        stride=args.stride,
        roi=tuple(float(part.strip()) for part in args.roi.split(",")),
        min_hand_ratio=args.min_hand_ratio,
        min_landmark_presence=args.min_landmark_presence,
        min_margin=args.min_margin,
        use_landmarks=not args.no_landmarks,
        letter_interval=args.letter_interval,
        speak_on_no_hand=not args.no_speech,
        mirror=not args.no_mirror,
        mirror_prediction=args.mirror_prediction,
        no_hand_frames=args.no_hand_frames,
        crop_padding=args.crop_padding,
        mic_index=args.mic_index,
        mic_timeout=args.mic_timeout,
        mic_phrase_time_limit=args.mic_phrase_time_limit,
        mic_ambient_duration=args.mic_ambient_duration,
        mic_energy_threshold=args.mic_energy_threshold if args.mic_energy_threshold is not None else 400.0,
        mic_dynamic_energy=args.mic_dynamic_energy,
    )
    inference.run_webcam()
