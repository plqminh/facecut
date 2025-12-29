import cv2
import numpy as np
import face_alignment
from ultralytics import YOLO
import mediapipe as mp
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch
import os
import urllib.request
import zipfile
import shutil

class VideoProcessor:
    def __init__(self, model_type='yolo'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        self.model = None
        self.detector = None
        self.mp_face_detection = None
        self.gender_net = None
        # InsightFace GenderAge Model: 0=Female, 1=Male
        self.gender_list = ['Female', 'Male'] 
        self.gender_model = "genderage.onnx"
        
        # Face Recognition
        self.rec_net = None
        self.rec_model = "w600k_mbf.onnx"
        self.reference_embedding = None
        
        self.ensure_models()
        self.load_models()
        
        print(f"Loading {self.model_type.upper()} model on {self.device}...")
        
        if self.model_type == 's3fd':
            # Initialize FaceAlignment with S3FD detector
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                                 face_detector='sfd', 
                                                 device=self.device)
            self.detector = self.fa.face_detector
        elif self.model_type == 'yolo':
            self.model = YOLO('yolo11n-face.pt')
            self.model.to(self.device)
        elif self.model_type == 'mediapipe':
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def ensure_models(self):
        # Gender
        if not os.path.exists(self.gender_model):
            print("Downloading gender model (ONNX)...")
            url = "https://huggingface.co/lithiumice/insightface/resolve/main/models/buffalo_l/genderage.onnx"
            try:
                urllib.request.urlretrieve(url, self.gender_model)
            except Exception as e:
                print(f"Failed to download gender model: {e}")

        # Rec
        if not os.path.exists(self.rec_model):
            print("Downloading face recognition model (ONNX)...")
            zip_name = "buffalo_s.zip"
            url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
            try:
                # Download Zip
                urllib.request.urlretrieve(url, zip_name)
                # Extract
                with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                    # Look for w600k_mbf.onnx inside
                    found = False
                    for file in zip_ref.namelist():
                        if file.endswith("w600k_mbf.onnx"):
                            source = zip_ref.open(file)
                            target = open(self.rec_model, "wb")
                            with source, target:
                                shutil.copyfileobj(source, target)
                            found = True
                            print(f"Extracted {self.rec_model}")
                            break
                    if not found:
                        print("w600k_mbf.onnx not found in zip!")
                
                # Cleanup
                os.remove(zip_name)
            except Exception as e:
                print(f"Failed to download/extract rec model: {e}")

    def load_models(self):
        # Gender
        try:
            self.gender_net = cv2.dnn.readNetFromONNX(self.gender_model)
            self.gender_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.gender_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            print(f"Failed to load gender model: {e}")
            self.gender_net = None

        # Rec
        try:
            self.rec_net = cv2.dnn.readNetFromONNX(self.rec_model)
            self.rec_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.rec_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except Exception as e:
            print(f"Failed to load rec model: {e}")
            self.rec_net = None

    def get_embedding(self, face_img):
        if self.rec_net is None:
            return None
            
        # Preprocessing for InsightFace Rec (MobileFaceNet/ArcFace)
        # 112x112, RGB
        # Normalization: (x - 127.5) / 128.0
        blob = cv2.dnn.blobFromImage(face_img, 1.0/127.5, (112, 112), 
                                   (127.5, 127.5, 127.5), 
                                   swapRB=True, crop=False)
        self.rec_net.setInput(blob)
        embedding = self.rec_net.forward()
        # Normalize the embedding vector
        norm_embedding = embedding / np.linalg.norm(embedding)
        return norm_embedding

    def compute_sim(self, feat1, feat2):
        if feat1 is None or feat2 is None:
            return 0.0
        return np.dot(feat1, feat2.T)[0][0]

    def set_reference_face(self, image_path):
        if not os.path.exists(image_path):
            return False, "File not found"
            
        img = cv2.imread(image_path)
        if img is None:
            return False, "Could not read image"
            
        # Detect face in reference image - use our own scan logic but simplified
        # For simplicity, use the configured detector.
        # But we need to define 'detect_faces' expects a detector...
        # We can just use the current initialized detector logic
        detections = self.detect_faces(img, min_conf=0.5)
        
        if not detections:
             return False, "No face found in reference image"
        
        # Pick largest face
        best_face = None
        max_area = 0
        img_h, img_w, _ = img.shape
        
        for x1, y1, x2, y2, conf in detections:
            area = (x2-x1) * (y2-y1)
            if area > max_area:
                max_area = area
                best_face = (x1, y1, x2, y2)
        
        if best_face:
            x1, y1, x2, y2 = best_face
            face_img = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
            if face_img.size > 0:
                self.reference_embedding = self.get_embedding(face_img)
                return True, "Reference face set"
                
        return False, "Could not process reference face"

    def predict_gender(self, face_img):
        if self.gender_net is None:
            return "Unknown"
        
        # InsightFace GenderAge Preprocessing
        # Resize to 96x96
        # Mean 127.5, Scale 1/128.0 (approx 0.0078125)
        # RGB (swapRB=True because OpenCV is BGR)
        blob = cv2.dnn.blobFromImage(face_img, 1.0/128.0, (96, 96), 
                                   (127.5, 127.5, 127.5), 
                                   swapRB=True, crop=False)
        self.gender_net.setInput(blob)
        preds = self.gender_net.forward()
        # Output shape is (1, 3). [0, 1] are gender logits.
        # 0 -> Female, 1 -> Male
        gender_idx = np.argmax(preds[0][:2])
        return self.gender_list[gender_idx]

    def check_obstruction(self, box, img_w, img_h, margin_ratio=0.0, max_obstruction=0.0):
        """
        Check if the bounding box is significantly obstructed by the frame edge.
        Box format: x1, y1, x2, y2
        margin_ratio: float 0.0-1.0, fraction of dimension to use as margin.
        max_obstruction: float 0.0-1.0, max allowed obstructed ratio.
        Returns True if obstructed (>max_obstruction), False otherwise.
        """
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1
        box_area = box_w * box_h
        
        if box_area <= 0:
            return True

        # Define Safe Zone
        margin_w = int(img_w * margin_ratio)
        margin_h = int(img_h * margin_ratio)
        
        safe_x1 = margin_w
        safe_y1 = margin_h
        safe_x2 = img_w - margin_w
        safe_y2 = img_h - margin_h
        
        # Calculate Intersection with Safe Zone
        inter_x1 = max(x1, safe_x1)
        inter_y1 = max(y1, safe_y1)
        inter_x2 = min(x2, safe_x2)
        inter_y2 = min(y2, safe_y2)
        
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        # Calculate Obstruction Ratio
        # Obstruction is the part of the box NOT in the intersection
        obstructed_area = box_area - inter_area
        obstruction_ratio = obstructed_area / box_area
        
        # Strictness check
        if obstruction_ratio > max_obstruction:
            return True
            
        return False

    def estimate_pose(self, detection, w, h):
        """
        Estimate face pose (yaw, pitch) from MediaPipe detection.
        Returns: (yaw, pitch) in degrees.
        """
        # MediaPipe Face Detection provides 6 keypoints:
        # 0: Right Eye
        # 1: Left Eye
        # 2: Nose Tip
        # 3: Mouth Center
        # 4: Right Ear Tragion
        # 5: Left Ear Tragion
        
        kps = detection.location_data.relative_keypoints
        
        def get_pt(idx):
            return np.array([kps[idx].x * w, kps[idx].y * h])

        re = get_pt(0) # Right Eye
        le = get_pt(1) # Left Eye
        nt = get_pt(2) # Nose Tip
        mc = get_pt(3) # Mouth Center
        ret = get_pt(4) # Right Ear
        let = get_pt(5) # Left Ear
        
        # Yaw Estimation
        # Compare distance from nose to left/right ears
        dist_n_re = np.linalg.norm(nt - ret)
        dist_n_le = np.linalg.norm(nt - let)
        
        # Avoid division by zero
        total_dist = dist_n_re + dist_n_le
        if total_dist == 0:
            return 0, 0
            
        # Ratio: 0.5 is center. <0.5 looking right (nose closer to right ear), >0.5 looking left
        yaw_ratio = dist_n_re / total_dist
        
        # Map ratio to degrees (approximate)
        # 0.5 -> 0 deg
        # 0.0 -> -90 deg (looking right)
        # 1.0 -> 90 deg (looking left)
        yaw = (yaw_ratio - 0.5) * 180
        
        # Pitch Estimation
        # Compare nose vertical position relative to eyes and mouth
        eye_mid = (re + le) / 2
        mouth_mid = mc
        
        dist_n_e = np.linalg.norm(nt - eye_mid)
        dist_n_m = np.linalg.norm(nt - mouth_mid)
        
        total_h = dist_n_e + dist_n_m
        if total_h == 0:
            return 0, 0
            
        # Ratio: 0.5 is roughly center (nose in middle)
        # Note: This is very rough. Nose is usually closer to eyes.
        pitch_ratio = dist_n_e / total_h
        
        # Calibrate center (empirically, nose is about 40% down from eyes to mouth)
        center_ratio = 0.4
        pitch = (pitch_ratio - center_ratio) * 180 
        
        return yaw, pitch

    def detect_faces(self, frame, min_conf, max_angle=90):
        """
        Unified detection method.
        Returns list of (x1, y1, x2, y2, conf)
        """
        detections = []
        
        if self.model_type == 's3fd':
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            preds = self.detector.detect_from_image(rgb_frame)
            if preds is not None:
                for pred in preds:
                    x1, y1, x2, y2, conf = pred
                    if conf >= min_conf:
                        detections.append((int(x1), int(y1), int(x2), int(y2), conf))
                        
        elif self.model_type == 'yolo':
            results = self.model(frame, verbose=False, conf=min_conf)
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box[:4])
                        conf = box[4]
                        detections.append((x1, y1, x2, y2, conf))
        
        elif self.model_type == 'mediapipe':
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_face_detection.process(rgb_frame)
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    conf = detection.score[0]
                    if conf >= min_conf:
                        # Check Angle
                        yaw, pitch = self.estimate_pose(detection, w, h)
                        if abs(yaw) > max_angle or abs(pitch) > max_angle:
                            continue
                            
                        bboxC = detection.location_data.relative_bounding_box
                        x1 = int(bboxC.xmin * w)
                        y1 = int(bboxC.ymin * h)
                        w_box = int(bboxC.width * w)
                        h_box = int(bboxC.height * h)
                        x2 = x1 + w_box
                        y2 = y1 + h_box
                        detections.append((x1, y1, x2, y2, conf))
                        
        return detections

    def scan_video(self, video_path, min_conf, obstruction_threshold=0.0, obstruction_margin=0.0, min_duration=0.0, max_angle=90, target_gender="All", rec_threshold=0.5, progress_callback=None, preview_callback=None, stop_event=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        valid_frames = []
        frame_idx = 0
        
        while cap.isOpened():
            if stop_event and stop_event.is_set():
                cap.release()
                return None, "Scanning stopped by user."

            ret, frame = cap.read()
            if not ret:
                break

            if progress_callback and frame_idx % 10 == 0:
                progress_callback(frame_idx / total_frames)

            # Run inference
            detections = self.detect_faces(frame, min_conf, max_angle)
            
            is_valid = False
            img_h, img_w, _ = frame.shape
            
            for x1, y1, x2, y2, conf in detections:
                face_passed = False
                # Obstruction Check
                if not self.check_obstruction((x1, y1, x2, y2), img_w, img_h, obstruction_margin, obstruction_threshold):
                    face_passed = True
                    
                # Check Gender if needed (and if face passed obstruction check)
                if face_passed and target_gender != "All":
                    # Crop face
                    face_img = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    if face_img.size > 0:
                        gender = self.predict_gender(face_img)
                        if gender != target_gender:
                            face_passed = False
                    else:
                        face_passed = False

                # Check Recognition if active
                if face_passed and self.reference_embedding is not None:
                    face_img = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    if face_img.size > 0:
                        emb = self.get_embedding(face_img)
                        sim = self.compute_sim(emb, self.reference_embedding)
                        if sim < rec_threshold:
                            face_passed = False
                    else:
                        face_passed = False

                if face_passed:
                    is_valid = True
                    break
            
            if is_valid:
                valid_frames.append(frame_idx)

            # Preview Callback
            if preview_callback and frame_idx % 2 == 0:
                annotated_frame = frame.copy()
                for x1, y1, x2, y2, conf in detections:
                    face_valid = True
                    if self.check_obstruction((x1, y1, x2, y2), img_w, img_h, obstruction_margin, obstruction_threshold):
                        face_valid = False
                    
                    gender_label = ""
                    if face_valid and self.reference_embedding is not None:
                         face_img = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                         if face_img.size > 0:
                            emb = self.get_embedding(face_img)
                            sim = self.compute_sim(emb, self.reference_embedding)
                            gender_label += f" Sim:{sim:.2f}"
                            if sim < rec_threshold:
                                face_valid = False

                    color = (0, 255, 0) if face_valid else (0, 0, 255)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"Conf: {conf:.2f}{gender_label}"
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                preview_callback(annotated_frame)
            
            frame_idx += 1

        cap.release()
        
        if not valid_frames:
            return [], "No valid frames found matching criteria."

        # Merge frames into clips
        gap_tolerance = int(fps * 0.5) 
        
        segments = []
        if valid_frames:
            start = valid_frames[0]
            prev = valid_frames[0]
            
            for f in valid_frames[1:]:
                if f - prev > gap_tolerance:
                    # Check duration
                    duration = (prev - start + 1) / fps
                    if duration >= min_duration:
                        segments.append({'start_frame': start, 'end_frame': prev})
                    start = f
                prev = f
            
            # Check last segment
            duration = (prev - start + 1) / fps
            if duration >= min_duration:
                segments.append({'start_frame': start, 'end_frame': prev})

        return segments, "Scanning complete."

    def render_video(self, video_path, output_path, segments, progress_callback=None, stop_event=None):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            original_clip = VideoFileClip(video_path)
            subclips = []
            
            total_segments = len(segments)
            for i, seg in enumerate(segments):
                if stop_event and stop_event.is_set():
                     original_clip.close()
                     return False, "Rendering stopped by user."
                
                start_frame = seg['start_frame']
                end_frame = seg['end_frame']
                
                start_time = start_frame / fps
                end_time = min((end_frame + 1) / fps, original_clip.duration)
                subclips.append(original_clip.subclip(start_time, end_time))
                
                if progress_callback:
                    progress_callback((i + 1) / total_segments)

            if not subclips:
                 original_clip.close()
                 return False, "No valid clips generated."

            final_clip = concatenate_videoclips(subclips)
            
            # Note: writing videofile is blocking and hard to report granular progress on without a custom logger
            # We can just let it run.
            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
            
            original_clip.close()
            final_clip.close()
            
            return True, "Processing complete."
        except Exception as e:
            return False, f"Video editing error: {str(e)}"

    def process_frame(self, frame, min_conf, obstruction_threshold=0.0, obstruction_margin=0.0, max_angle=90, target_gender="All", rec_threshold=0.5):
        """
        Process a single frame for preview.
        Returns: annotated_frame, is_valid_frame
        """
        detections = self.detect_faces(frame, min_conf, max_angle)
        
        annotated_frame = frame.copy()
        frame_valid = False
        img_h, img_w, _ = frame.shape
        
        for x1, y1, x2, y2, conf in detections:
            face_valid = True
            if self.check_obstruction((x1, y1, x2, y2), img_w, img_h, obstruction_margin, obstruction_threshold):
                face_valid = False
            
            gender_label = ""
            if face_valid and self.reference_embedding is not None:
                    face_img = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    if face_img.size > 0:
                        emb = self.get_embedding(face_img)
                        sim = self.compute_sim(emb, self.reference_embedding)
                        gender_label += f" Sim:{sim:.2f}"
                        if sim < rec_threshold:
                            face_valid = False

            if face_valid:
                frame_valid = True

            color = (0, 255, 0) if face_valid else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"Conf: {conf:.2f}{gender_label}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return annotated_frame, frame_valid
