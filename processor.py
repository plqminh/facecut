import cv2
import numpy as np
import face_alignment
from ultralytics import YOLO
import mediapipe as mp
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import os
import urllib.request
import zipfile
import shutil
import onnxruntime as ort

class VideoProcessor:
    def __init__(self, model_type='yolo'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_type = model_type
        self.model = None
        self.detector = None
        self.mp_face_detection = None
        self.gender_net = None
        # FairFace Model (ResNet34, trained on racially-balanced dataset)
        # Multi-head output: race(7), gender(2), age(9)
        # Gender: Index 0 = Male, Index 1 = Female
        self.gender_list = ['Male', 'Female'] 
        self.gender_model = "fairface.onnx"
        
        # Face Recognition
        self.rec_net = None
        self.rec_model = "w600k_mbf.onnx"
        self.reference_embedding = None
        
        # FAN (Face Alignment Network) for Strict Validation
        self.fan = None
        
        # Smoothing for quality scores (Exponential Moving Average)
        self.quality_ema = None
        self.quality_ema_alpha = 0.3  # Lower = smoother, Higher = more responsive
        
        self.ensure_models()
        self.load_models()
        
        print(f"Loading {self.model_type.upper()} model on {self.device}...")
        
        if self.model_type == 's3fd':
            # Initialize FaceAlignment with S3FD detector
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                                 face_detector='sfd', 
                                                 device=self.device,
                                                 compile=False)
            self.detector = self.fa.face_detector
        elif self.model_type == 'yolo':
            self.model = YOLO('yolo11n-face.pt')
            self.model.to(self.device)
        elif self.model_type == 'mediapipe':
            self.mp_face_detection = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.5)
        else:
            raise ValueError(f"Unknown model type: {model_type}")


    def ensure_fan(self):
        if self.fan is None:
            print(f"Loading FAN (Face Alignment Network) for strict validation on {self.device}...")
            try:
                # Use CPU/Cuda based on avail. '2D' landmarks.
                self.fan = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                                      device=self.device, 
                                                      face_detector='blazeface',
                                                      compile=False) # Use lightweight internal detector?
                                                      # Actually we pass the rect, so detector matters less, but blazeface is standard.
                print(f"FAN loaded on {self.device}.")
            except Exception as e:
                print(f"Failed to load FAN on {self.device}: {e}")
                
    def ensure_models(self):
        # Gender (FairFace - racially balanced)
        if not os.path.exists(self.gender_model):
            print("Downloading FairFace gender model (ONNX)...")
            url = "https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx"
            try:
                urllib.request.urlretrieve(url, self.gender_model)
                print(f"Downloaded {self.gender_model}")
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
        # Gender (FairFace via onnxruntime - multi-head output)
        try:
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                
            self.gender_net = ort.InferenceSession(
                self.gender_model, providers=providers)
            input_meta = self.gender_net.get_inputs()[0]
            self.gender_input_name = input_meta.name
            self.gender_output_names = [o.name for o in self.gender_net.get_outputs()]
            
            active_provider = self.gender_net.get_providers()[0]
            print(f"Loaded FairFace gender model ({len(self.gender_output_names)} heads) using {active_provider}")
        except Exception as e:
            print(f"Failed to load gender model: {e}")
            self.gender_net = None

        # Rec
        try:
            providers = ['CPUExecutionProvider']
            if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                
            self.rec_net = ort.InferenceSession(self.rec_model, providers=providers)
            self.rec_input_name = self.rec_net.get_inputs()[0].name
            self.rec_output_names = [o.name for o in self.rec_net.get_outputs()]
            
            active_provider = self.rec_net.get_providers()[0]
            print(f"Loaded Face Recognition model using {active_provider}")
        except Exception as e:
            print(f"Failed to load rec model: {e}")
            self.rec_net = None





    def get_embedding(self, face_img, landmarks=None, tilt_boost=1.0, apply_smoothing=True):
        if self.rec_net is None:
            return None, 0.0

        # Quality Metrics Calculation (Pre-normalization)
        
        # Alignment
        blob_input = face_img
        
        # Preprocessing for InsightFace Rec (MobileFaceNet/ArcFace)
        # 112x112, RGB
        # Normalization: (x - 127.5) / 128.0
        
        # If image is big, resize. If 112x112 (aligned), keep.
        if face_img.shape[0] != 112 or face_img.shape[1] != 112:
            # Standard Resize if not already aligned
            blob_input = cv2.resize(face_img, (112, 112))
        else:
            blob_input = face_img.copy()
             
        # Preprocessing for ONNX Runtime
        blob_input = cv2.cvtColor(blob_input, cv2.COLOR_BGR2RGB)
        blob_input = blob_input.astype(np.float32)
        blob_input = (blob_input - 127.5) / 127.5
        blob_input = np.transpose(blob_input, (2, 0, 1))
        blob_input = np.expand_dims(blob_input, axis=0)
        
        outputs = self.rec_net.run(self.rec_output_names, {self.rec_input_name: blob_input})
        embedding = outputs[0]
        # Quality Score (Feature Norm)
        feature_norm = np.linalg.norm(embedding)
        
        # Composite Quality Score (FAN ONLY Mode)
        # We rely solely on FAN validation (done in evaluate_face) + Raw Feature Norm.
        # No penalties for blur, brightness, or padding.
        composite_score = feature_norm
        
        # Apply exponential moving average smoothing to reduce fluctuations
        if apply_smoothing and composite_score > 0:
            if self.quality_ema is None:
                self.quality_ema = composite_score
            else:
                self.quality_ema = self.quality_ema_alpha * composite_score + \
                                   (1 - self.quality_ema_alpha) * self.quality_ema
            composite_score = self.quality_ema
        
        # Normalize the embedding vector
        norm_embedding = embedding / (feature_norm + 1e-5)
        
        return norm_embedding, composite_score

    def align_face(self, img, landmarks):
        """
        Align face using landmarks to standard ArcFace template (112x112).
        Landmarks: list/array of 5 points OR 68 points.
        Returns: aligned_img (112, 112, 3)
        """
        if landmarks is None:
            return None
            
        landmarks = np.array(landmarks)
        
        if len(landmarks) == 68:
            # Convert 68-point DLIB to 5-point ArcFace
            # 36-41: Right Eye, 42-47: Left Eye, 30: Nose, 48: R Mouth, 54: L Mouth
            # Note: FAN uses 0-indexed logic matching dlib
            
            # Left Eye (average of 36-41) -> Image coords
            # Wait, 36-41 is Right Eye in typical Medical terms (Patient Right), but Image Left?
            # Standard Dlib: 36..41 is LEFT EYE (User View Left, Patient Right). 42..47 is RIGHT EYE.
            # ArcFace expects: [RightEye, LeftEye, Nose, RightMouth, LeftMouth] (Image Coords)
            # Dlib 36 (outer left) -> 39 (inner left). 
            # Dlib 42 (inner right) -> 45 (outer right).
            
            # Let's verify standard 5 point:
            # Image Left Eye (Patient Right), Image Right Eye
            # ArcFace ref points look like: [38, 51] (Left/Right?), [73, 51] ...
            # 38 is x (Left), 73 is x (Right). So Index 0 is LEFT eye (Image Left).
            # WAIT. Let's check my 'detect_faces' MediaPipe mapping:
            # lms = [r_eye, l_eye, nose, r_mouth, l_mouth]
            # r_eye (Image Right, i.e. patient left) - NO.
            # MediaPipe 0=RightEye (Patient Right), 1=LeftEye (Patient Left).
            # "Right Eye" usually means Patient Right (Image Left).
            # Let's check coordinates.
            # src[0] = [38, 51] -> x=38 (Image Left). So Index 0 is IMAGE LEFT EYE.
            # src[1] = [73, 51] -> x=73 (Image Right). So Index 1 is IMAGE RIGHT EYE.
            
            # My MediaPipe map:
            # gp(0) is Right Eye (Patient Right / Image Left).
            # So my 5-point array is [ImageLeftEye, ImageRightEye, Nose, ImageLeftMouth, ImageRightMouth] ??
            # ArcFace src expects: [RightEye??, LeftEye??, ... ] 
            # Actually standard Insightface is:
            # 0: Left Eye (Image Left)
            # 1: Right Eye (Image Right)
            # 2: Nose
            # 3: Left Mouth Corner (Image Left)
            # 4: Right Mouth Corner (Image Right)
            
            # Let's map 68 points to this order:
            # 0 (ImgLeftEye): Mean(36...41)
            # 1 (ImgRightEye): Mean(42...47)
            # 2 (Nose): 30
            # 3 (ImgLeftMouth): 48 
            # 4 (ImgRightMouth): 54
            
            le_idxs = list(range(36, 42))
            re_idxs = list(range(42, 48))
            
            img_le = np.mean(landmarks[le_idxs], axis=0) # Image Left Eye
            img_re = np.mean(landmarks[re_idxs], axis=0) # Image Right Eye
            nose = landmarks[30]
            img_lm = landmarks[48] # Left Mouth Corner
            img_rm = landmarks[54] # Right Mouth Corner
            
            # Check my Processor 5-point order:
            # src array: [38, 51] (Left), [73, 51] (Right)
            # So yes, 0=Left, 1=Right.
            
            dst = np.array([img_le, img_re, nose, img_lm, img_rm], dtype=np.float32)

        elif len(landmarks) == 5:
            # Assumed order: [ImageLeftEye, ImageRightEye, Nose, ImageLeftMouth, ImageRightMouth]
            # Note: My Mediapipe wrapper had: [r_eye(0), l_eye(1)...]
            # MP 0 is Right Eye (Patient Right -> Image Left). Correct.
            dst = np.array(landmarks, dtype=np.float32)
        else:
            return None
            
        # Standard ArcFace 112x112 reference points
        src = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041] ], dtype=np.float32)
            
        # dst is already defined above as 5 points
        
        # Estimate affine transform
        try:
            tform = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)[0]
            if tform is None:
                # Fallback to simple affine if LMEDS fails
                tform = cv2.estimateAffinePartial2D(dst, src)[0]
        except Exception as e:
            print(f"Align Error: {e}")
            return None
             
        if tform is None:
            return None
            
        aligned_img = cv2.warpAffine(img, tform, (112, 112), flags=cv2.INTER_CUBIC, borderValue=0.0)
        return aligned_img

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
        
        for x1, y1, x2, y2, conf, lm in detections:
            area = (x2-x1) * (y2-y1)
            if area > max_area:
                max_area = area
                best_face = (x1, y1, x2, y2, conf, lm)
        
        if best_face:
            x1, y1, x2, y2, _, landmarks = best_face
            # If landmarks exist, align from FULL image
            if landmarks is not None:
                face_img = self.align_face(img, landmarks)
                if face_img is None:
                    face_img = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
            else:
                face_img = img[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
            
            if face_img.size > 0:
                # Disable smoothing for reference face to get raw embedding
                self.reference_embedding, _ = self.get_embedding(face_img, apply_smoothing=False)
                return True, "Reference face set"
                
        return False, "Could not process reference face"

    def predict_gender(self, frame, bbox, aligned_img=None):
        """
        Gender prediction using FairFace (ResNet34, racially-balanced training).
        
        FairFace was specifically designed for fair face attribute prediction
        across 7 race groups including East Asian and Southeast Asian.
        
        Args:
            frame: Full BGR frame
            bbox: (x1, y1, x2, y2) face bounding box
            aligned_img: Unused, kept for API compat
        
        FairFace output: 3 heads [race_logits(7), gender_logits(2), age_logits(9)]
        Gender: Index 0 = Male, Index 1 = Female
        """
        if self.gender_net is None:
            return "Unknown"
        
        if frame is None or frame.size == 0:
            return "Unknown"
        
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return "Unknown"
        
        def preprocess_fairface(img, face_bbox=None):
            """FairFace preprocessing: 25% padded crop, 224x224, ImageNet normalization."""
            if face_bbox is not None:
                bx1, by1, bx2, by2 = face_bbox
                w, h = bx2 - bx1, by2 - by1
                padding = 0.25
                x_pad = int(w * padding)
                y_pad = int(h * padding)
                bx1 = max(0, bx1 - x_pad)
                by1 = max(0, by1 - y_pad)
                bx2 = min(img.shape[1], bx2 + x_pad)
                by2 = min(img.shape[0], by2 + y_pad)
                img = img[by1:by2, bx1:bx2]
            
            if img.size == 0:
                return None
            
            # Resize to 224x224
            img = cv2.resize(img, (224, 224))
            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # ImageNet normalization
            img = img.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img = (img - mean) / std
            # HWC -> CHW, add batch dim
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return img
        
        try:
            # Preprocess with padded bbox crop
            blob = preprocess_fairface(frame, (x1, y1, x2, y2))
            if blob is None:
                return "Unknown"
            
            # Run inference (3 heads: race, gender, age)
            outputs = self.gender_net.run(
                self.gender_output_names, {self.gender_input_name: blob})
            
            # Gender is the second head (index 1)
            gender_logits = outputs[1][0]
            
            # Softmax
            exp_logits = np.exp(gender_logits - np.max(gender_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            gender_idx = int(np.argmax(probs))
            confidence = probs[gender_idx]
            
            # If confidence is too low, return Unknown
            if confidence < 0.55:
                return "Unknown"
            
            return self.gender_list[gender_idx]
            
        except Exception as e:
            print(f"Gender prediction error: {e}")
            return "Unknown"



    def estimate_pose_from_landmarks(self, landmarks):
        """
        Estimate Yaw, Pitch, Roll from 5 landmarks OR 68-point landmarks.
        Landmarks: [RightEye, LeftEye, Nose, RightMouth, LeftMouth] (Image coords)
                   OR 68-point array.
        Returns: yaw, pitch, roll (degrees)
        """
        landmarks = np.array(landmarks)
        
        if len(landmarks) == 68:
             # Extract 5 points from 68 for pose estimation
             le_idxs = list(range(36, 42))
             re_idxs = list(range(42, 48))
             
             img_le = np.mean(landmarks[le_idxs], axis=0)
             img_re = np.mean(landmarks[re_idxs], axis=0)
             nose = landmarks[30]
             img_lm = landmarks[48]
             img_rm = landmarks[54]
             
             # Convert to 5-point standard array: [RE, LE, N, RM, LM]
             # Note: 'RE' in standard pose estimator usually means Image Left (Patient Right)?
             # Check logic below:
             # It uses P3P solver. It expects 3D model points.
             # self.model_points correspond to:
             # 0: Nose (0.0, 0.0, 0.0) -> Index 2
             # 1: Chin (0.0, -330.0, -65.0) -> Not in 5 point
             # 2: Left Eye (-225.0, 170.0, -135.0) -> Index 1
             # 3: Right Eye (225.0, 170.0, -135.0) -> Index 0
             # 4: Left Mouth (-150.0, -150.0, -125.0) -> Index 4
             # 5: Right Mouth (150.0, -150.0, -125.0) -> Index 3
             
             # So we need to feed: [RightEye, LeftEye, Nose, RightMouth, LeftMouth]
             # My variables: img_re (Image Right / Patient Left), img_le (Image Left / Patient Right)
             # Wait, my dlib logic: 
             # le_idxs (36-41) -> Image Left (Patient Right)
             # re_idxs (42-47) -> Image Right (Patient Left)
             
             # If pose model expects Index 0 = Right Eye (225.0 ...).
             # +X is usually Right.
             # So Index 0 is Image Right (Patient Left)?
             
             # Let's verify standard heuristics.
             # re = landmarks[0]. le = landmarks[1].
             # code: dx = le[0] - re[0]. 
             # If expected roll 0: dx should be positive (le.x > re.x).
             # So le must be Image Right. re must be Image Left.
             # My variables: img_le = Image Left (Patient Right). img_re = Image Right (Patient Left).
             # So re(index 0) should be img_le. le(index 1) should be img_re.
             
             # Order: [ImageLeft, ImageRight, Nose, ImageLeftMouth, ImageRightMouth]
             lms_5 = np.array([img_le, img_re, nose, img_lm, img_rm], dtype=np.float32)
             landmarks = lms_5
        elif landmarks is None or len(landmarks) != 5:
            return 0, 0, 0
            
        re = landmarks[0]
        le = landmarks[1]
        nose = landmarks[2]
        rm = landmarks[3]
        lm = landmarks[4]
        
        # Roll: Angle of eye line
        dx = le[0] - re[0]
        dy = le[1] - re[1]
        roll = np.degrees(np.arctan2(dy, dx))
        
        # Normalize points by rotating -roll (make eyes horizontal)
        center = nose
        M = cv2.getRotationMatrix2D((center[0], center[1]), roll, 1.0)
        pts = np.array([re, le, nose, rm, lm]).reshape(-1, 1, 2)
        pts_rot = cv2.transform(pts, M).squeeze()
        
        tre = pts_rot[0]
        tle = pts_rot[1]
        tn = pts_rot[2]
        trm = pts_rot[3]
        tlm = pts_rot[4]
        
        # Yaw: Nose deviation from eye midpoint (Horizontal)
        eye_mid_x = (tre[0] + tle[0]) / 2
        eye_width = tle[0] - tre[0]
        # Nose off-center ratio. 
        # Factor ~300 found empirically for approximate degrees? 
        # Actually simpler: if nose is at eye, yaw is ~90.
        # nose_off / (eye_width/2) = 1.0 => 90 deg?
        if eye_width > 1e-5:
            yaw = ((tn[0] - eye_mid_x) / (eye_width / 2)) * 60 # approx deg?
        else:
            yaw = 0
            
        # Pitch: Nose vertical position
        # Eye mid Y
        eye_mid_y = (tre[1] + tle[1]) / 2
        mouth_mid_y = (trm[1] + tlm[1]) / 2
        total_h = mouth_mid_y - eye_mid_y
        nose_h = tn[1] - eye_mid_y
        
        if total_h > 1e-5:
            ratio = nose_h / total_h
            # Standard ratio is approx 0.35-0.4?
            pitch = (ratio - 0.38) * 150 # scale factor
        else:
            pitch = 0
            
        return yaw, pitch, roll

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

    def rotate_coords(self, coords, landmarks, rotation, w, h):
        """
        Map coordinates from a rotated frame back to original.
        rotation: cv2.ROTATE_90_CLOCKWISE or cv2.ROTATE_90_COUNTERCLOCKWISE
        w, h: Dimensions of the ORIGINAL frame (before rotation).
        coords: (x1, y1, x2, y2)
        landmarks: list of [x, y] or None
        """
        x1, y1, x2, y2 = coords
        
        def transform_point(pt):
            x, y = pt
            if rotation == cv2.ROTATE_90_CLOCKWISE:
                # 90 CW: x' = h - 1 - y, y' = x
                # Inverse: x = y', y = h - 1 - x'
                # Here pt is (x', y') in rotated frame
                # The rotated frame has width=h, height=w.
                # FORMULA MAPS BACK TO ORIGINAL (w, h)
                # x_orig = y_rot
                # y_orig = h - 1 - x_rot
                return [y, h - 1 - x]
            elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
                # 90 CCW: x' = y, y' = w - 1 - x
                # Inverse: x = w - 1 - y', y = x'
                # x_orig = w - 1 - y_rot
                # y_orig = x_rot
                return [w - 1 - y, x]
            return [x, y]

        p1 = transform_point((x1, y1))
        p2 = transform_point((x2, y2))
        p3 = transform_point((x1, y2))
        p4 = transform_point((x2, y1))
        
        xs = [p1[0], p2[0], p3[0], p4[0]]
        ys = [p1[1], p2[1], p3[1], p4[1]]
        
        lx1, lx2 = min(xs), max(xs)
        ly1, ly2 = min(ys), max(ys)
        
        new_lms = None
        if landmarks is not None:
             new_lms = []
             for lm in landmarks:
                  new_lms.append(transform_point(lm))
             new_lms = np.array(new_lms)
             
        return (int(lx1), int(ly1), int(lx2), int(ly2)), new_lms

    def detect_faces(self, frame, min_conf, max_angle=90):
        """
        Multi-rotation wrapper for detection.
        Optimized to minimize color space conversions.
        """
        # Convert to RGB once for all detection passes
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Pass 1: Original
        detections = self._detect_single_pass(frame, rgb_frame, min_conf, max_angle)
        
        if max_angle > 60 and len(detections) == 0:
             h, w = frame.shape[:2]
             
             # Pass 2: 90 CW
             frame_90 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
             rgb_90 = cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE)
             dets_90 = self._detect_single_pass(frame_90, rgb_90, min_conf, max_angle)
             
             for det in dets_90:
                  x1, y1, x2, y2, conf, lms = det
                  (rx1, ry1, rx2, ry2), rlms = self.rotate_coords((x1, y1, x2, y2), lms, cv2.ROTATE_90_CLOCKWISE, w, h)
                  detections.append((rx1, ry1, rx2, ry2, conf, rlms))
                  
             # Pass 3: 90 CCW
             frame_n90 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
             rgb_n90 = cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
             dets_n90 = self._detect_single_pass(frame_n90, rgb_n90, min_conf, max_angle)
             
             for det in dets_n90:
                  x1, y1, x2, y2, conf, lms = det
                  (rx1, ry1, rx2, ry2), rlms = self.rotate_coords((x1, y1, x2, y2), lms, cv2.ROTATE_90_COUNTERCLOCKWISE, w, h)
                  detections.append((rx1, ry1, rx2, ry2, conf, rlms))
             
             # NMS to merge duplicates
             if len(detections) > 1:
                  boxes = []
                  scores = []
                  for d in detections:
                       boxes.append([d[0], d[1], d[2]-d[0], d[3]-d[1]]) # x, y, w, h
                       scores.append(float(d[4]))
                  
                  indices = cv2.dnn.NMSBoxes(boxes, scores, min_conf, 0.4)
                  if len(indices) > 0:
                       new_dets = []
                       for i in indices.flatten():
                            new_dets.append(detections[i])
                       detections = new_dets
        
        return detections

    def _detect_single_pass(self, frame, rgb_frame, min_conf, max_angle=90):
        """
        Unified detection method.
        Returns list of (x1, y1, x2, y2, conf, landmarks)
        landmarks: list of 5 (x,y) tuples or None.
        """
        detections = []
        
        if self.model_type == 's3fd':
            preds = self.detector.detect_from_image(rgb_frame)
            if preds is not None:
                for pred in preds:
                    x1, y1, x2, y2, conf = pred
                    if conf >= min_conf:
                        detections.append((int(x1), int(y1), int(x2), int(y2), conf, None))
                        
        elif self.model_type == 'yolo':
            results = self.model(frame, verbose=False, conf=min_conf)
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.data.cpu().numpy()
                    
                    # Keypoints
                    keypoints = None
                    if hasattr(result, 'keypoints') and result.keypoints is not None:
                         # Shape: [N, 5, 2] or [N, 5, 3] (conf)
                         # We want [N, 5, 2]
                         kps = result.keypoints.xy.cpu().numpy()
                    
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box[:4])
                        conf = box[4]
                        
                        lms = None
                        if keypoints is not None and i < len(keypoints):
                             lms = keypoints[i] # 5 points (Ref: RightEye, LeftEye, Nose, RightMouth, LeftMouth)
                        
                        # Pose Filter for YOLO - Very lenient to allow extreme angles
                        if lms is not None:
                             yaw, pitch, roll = self.estimate_pose_from_landmarks(lms)
                             # Only filter extremely unrealistic poses (likely false positives)
                             # Allow up to ~135 degrees for real profile/extreme views
                             if abs(yaw) > 135 and abs(pitch) > 135:
                                  continue
                             # Filter only extreme profile views beyond physical possibility
                             if abs(yaw) > 150:
                                  continue
                             
                        detections.append((x1, y1, x2, y2, conf, lms))
        
        elif self.model_type == 'mediapipe':
            results = self.mp_face_detection.process(rgb_frame)
            if results.detections:
                h, w, _ = frame.shape
                for detection in results.detections:
                    conf = detection.score[0]
                    if conf >= min_conf:
                        bboxC = detection.location_data.relative_bounding_box
                        x1 = int(bboxC.xmin * w)
                        y1 = int(bboxC.ymin * h)
                        w_box = int(bboxC.width * w)
                        h_box = int(bboxC.height * h)
                        x2 = x1 + w_box
                        y2 = y1 + h_box
                        
                        # Landmarks
                        kps = detection.location_data.relative_keypoints
                        def gp(i): return [kps[i].x * w, kps[i].y * h]
                        
                        l_eye = gp(1)
                        r_eye = gp(0)
                        nose = gp(2)
                        mouth = gp(3)
                        
                        # Fake mouth corners for 5-point
                        d_eyes = np.linalg.norm(np.array(l_eye) - np.array(r_eye))
                        l_mouth = [mouth[0] - d_eyes*0.25, mouth[1] + d_eyes*0.1]
                        r_mouth = [mouth[0] + d_eyes*0.25, mouth[1] + d_eyes*0.1]
                        
                        lms = np.array([r_eye, l_eye, nose, r_mouth, l_mouth]) # Order: RE, LE, N, RM, LM
                        
                        # Check Angle with new estimator - allow extreme rotated faces
                        yaw, pitch, roll = self.estimate_pose_from_landmarks(lms)
                        # Only filter extremely unrealistic poses (likely false positives)
                        # Allow up to ~135 degrees for real profile/extreme views
                        if abs(yaw) > 135 and abs(pitch) > 135:
                             continue
                        # Filter only extreme profile views beyond physical possibility
                        if abs(yaw) > 150:
                             continue
                        
                        detections.append((x1, y1, x2, y2, conf, lms))
                        
        return detections



    def evaluate_face(self, frame, detection, target_gender, rec_threshold, min_face_quality, force_tilt_boost=False, rgb_frame=None):
        """
        Evaluate a single face against criteria.
        Returns: (passed_filters, low_quality_fail_only, details_dict)
        low_quality_fail_only: True if face failed ONLY due to quality score.
        details_dict includes 'gender' key with the raw predicted gender string.
        """
        x1, y1, x2, y2, conf, landmarks = detection
        img_h, img_w = frame.shape[:2]
        
        # FAN Enhancement (Strict Mode)
        if min_face_quality > 0:
             # 1. Validation: Use FAN to confirm it's a valid face and refine landmarks
             self.ensure_fan()
             # We need to crop roughly to feed FAN detection with existing box hint.
             # Actually FAN takes image and face_rects.
             # FaceAlignment.get_landmarks(image, detected_faces=[(x1, y1, x2, y2)])
             
             try:
                  # Use pre-converted RGB frame if available for performance
                  if rgb_frame is None:
                      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  
                  preds = self.fan.get_landmarks(rgb_frame, detected_faces=[(x1, y1, x2, y2)])
                  
                  if preds is None or len(preds) == 0:
                       # FAN Failed -> Bad Face Structure
                       return False, False, {"label": " NoStruct", "quality": 0.0, "gender": "Unknown"}
                  
                  # FAN Succeeded -> Use refined landmarks
                  landmarks = preds[0] # 68 points
                  
             except Exception as e:
                  print(f"FAN Error: {e}")
                  # Fallback or strict fail? Let's strict fail to be safe.
                  return False, False, {"label": " FanErr", "quality": 0.0, "gender": "Unknown"}
        
        passed = True
        low_quality_fail = False
        gender_label = ""
        quality_score = 0.0
        rec_score = 0.0
        detected_gender = "Unknown"
        
        # Pre-compute face crop and aligned face for both gender and quality checks
        face_crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
        align_img = None
        if landmarks is not None:
             align_img = self.align_face(frame, landmarks)
        
        # Always predict gender for tagging (even when target_gender == "All")
        if face_crop.size > 0:
             detected_gender = self.predict_gender(frame, (x1, y1, x2, y2))
             gender_label += f" {detected_gender}"
        
        # Gender filter check
        if target_gender != "All":
             if face_crop.size > 0:
                  if detected_gender == "Unknown":
                       # Uncertain detection - skip this face when filtering by gender
                       passed = False
                  elif detected_gender != target_gender:
                       passed = False
             else:
                  passed = False
                   
        if not passed:
             return False, False, {"label": gender_label, "quality": 0.0, "gender": detected_gender}

        # Rec / Quality
        if min_face_quality > 0 or self.reference_embedding is not None:
             # align_img already computed above
             use_img = align_img if align_img is not None else face_crop
             
             if use_img is None or use_img.size == 0:
                 use_img = face_crop
                 
             if use_img.size > 0:
                # Determine if we need to compute embedding
                 # Case 1: Recognition is enabled (reference_embedding set) -> ALWAYS need embedding
                 # Case 2: Recognition disabled, but Quality check needed -> Need embedding
                 need_embedding = (self.reference_embedding is not None) or (min_face_quality > 0)
                 
                 if need_embedding:
                     # Tilt Boost Calculation
                     tilt_boost = 1.0
                     should_boost = force_tilt_boost
                     
                     if should_boost:
                          tilt_boost = 1.3
                               
                     emb, quality = self.get_embedding(use_img, tilt_boost=tilt_boost)
                     quality_score = quality
                     
                     # Check Quality
                     if min_face_quality > 0:
                         gender_label += f" Q:{quality:.1f}"
                         if quality < min_face_quality:
                             passed = False
                             low_quality_fail = True 
                             
                     # Check Rec
                     if passed and self.reference_embedding is not None:
                          sim = self.compute_sim(emb, self.reference_embedding)
                          rec_score = sim
                          gender_label += f" Sim:{sim:.2f}"
                          if sim < rec_threshold:
                               passed = False
                               low_quality_fail = False # Failed Rec
                 
             else:
                 passed = False
        
        return passed, low_quality_fail, {"label": gender_label, "quality": quality_score, "gender": detected_gender}

    def scan_video(self, video_path, min_conf, min_duration=0.0, max_angle=90, target_gender="All", rec_threshold=0.5, min_face_quality=0.0, progress_callback=None, preview_callback=None, stop_event=None, start_time=0.0, end_time=0.0, keep_no_face=True, min_no_face_duration=0.0):
        # Reset quality EMA at the start of each video scan for fresh smoothing
        self.quality_ema = None
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate start frame from start_time
        start_frame = int(start_time * fps) if start_time > 0 else 0
        start_frame = min(start_frame, total_frames - 1)  # Clamp to valid range
        
        # Calculate end frame from end_time (0 means process to the end)
        if end_time > 0:
            end_frame = int(end_time * fps)
            end_frame = min(end_frame, total_frames)  # Clamp to valid range
        else:
            end_frame = total_frames
        
        # Ensure end_frame > start_frame
        if end_frame <= start_frame:
            end_frame = total_frames
        
        # Seek to start position if needed
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        valid_frames = []
        # Track per-frame gender detections: frame_idx -> set of genders
        frame_genders = {}
        # Track which frames are no-face (for min_no_face_duration filtering)
        no_face_frames = set()
        frame_idx = start_frame
        
        while cap.isOpened():
            if stop_event and stop_event.is_set():
                cap.release()
                return None, "Scanning stopped by user."

            # Stop if we've reached the end frame
            if frame_idx >= end_frame:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if progress_callback and frame_idx % 10 == 0:
                # Calculate progress based on frames in the specified range
                frames_to_process = end_frame - start_frame
                frames_processed = frame_idx - start_frame
                progress = frames_processed / frames_to_process if frames_to_process > 0 else 1.0
                progress_callback(progress)

            # Run inference - convert to RGB once and reuse for detection and evaluation
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detect_faces(frame, min_conf, max_angle)
            
            is_valid = False
            has_face_with_low_quality = False
            img_h, img_w, _ = frame.shape
            
            low_qual_candidates = 0
            frame_gender_set = set()
            
            # Phase 1: Standard Evaluation
            evaluated_detections = [] # Store (det, details, passed) for preview
            
            if len(detections) == 0:
                # No face detected in this frame
                if keep_no_face:
                    is_valid = True
                else:
                    is_valid = False
            else:
                for det in detections:
                     passed, low_qual_fail, details = self.evaluate_face(frame, det, target_gender, rec_threshold, min_face_quality, rgb_frame=rgb_frame)
                     if passed:
                          is_valid = True
                     if low_qual_fail:
                          low_qual_candidates += 1
                          has_face_with_low_quality = True
                     
                     # Track gender from this detection
                     det_gender = details.get("gender", "Unknown")
                     if det_gender and det_gender != "Unknown":
                          frame_gender_set.add(det_gender)
                     
                     evaluated_detections.append((det, passed, details))

                # If no face passed filters AND at least one failed due to low quality,
                # then this frame should be discarded (face present but bad quality).
                # If faces failed for OTHER reasons (wrong gender, wrong identity),
                # still treat as no-valid-face frame -> keep it.
                if not is_valid:
                    if has_face_with_low_quality:
                        # Face detected but quality too low -> discard frame
                        is_valid = False
                    else:
                        # Face detected but failed other filters (gender/rec) ->
                        # keep the frame (treat like no relevant face)
                        is_valid = True

                # Phase 2: Smart Rotation Retry (only if face with low quality was found)
                if not is_valid and low_qual_candidates > 0 and max_angle > 60 and min_face_quality > 0:
                     # Retry with rotations
                     h, w = frame.shape[:2]
                     rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
                     
                     for rot in rotations:
                          frame_rot = cv2.rotate(frame, rot)
                          rgb_rot = cv2.rotate(rgb_frame, rot)
                          dets_rot = self._detect_single_pass(frame_rot, rgb_rot, min_conf, max_angle)
                          
                          for d_rot in dets_rot:
                               # Map back
                               x1, y1, x2, y2, conf, lms = d_rot
                               (rx1, ry1, rx2, ry2), rlms = self.rotate_coords((x1, y1, x2, y2), lms, rot, w, h)
                               det_orig = (rx1, ry1, rx2, ry2, conf, rlms)
                               
                               # Evaluate on the ROTATED frame (where face is upright)
                               passed, low_qual_fail, details = self.evaluate_face(frame_rot, d_rot, target_gender, rec_threshold, min_face_quality, force_tilt_boost=True, rgb_frame=rgb_rot)
                               
                               if passed:
                                    is_valid = True
                                    evaluated_detections.append((det_orig, passed, details))
                                    detections.append(det_orig)
                                    # Track gender from rotation retry
                                    det_gender = details.get("gender", "Unknown")
                                    if det_gender and det_gender != "Unknown":
                                         frame_gender_set.add(det_gender)
                                    break
                          
                          if is_valid:
                               break
            
            if is_valid:
                valid_frames.append(frame_idx)
                if frame_gender_set:
                    frame_genders[frame_idx] = frame_gender_set
                if len(detections) == 0:
                    no_face_frames.add(frame_idx)

            # Preview Callback
            if preview_callback and frame_idx % 2 == 0:
                annotated_frame = frame.copy()
                
                # If we succeeded in smart rotation, hide failed detections.
                final_detections_to_draw = evaluated_detections
                if is_valid and low_qual_candidates > 0:
                     # Keep only valid ones
                     valid_only = [d for d in evaluated_detections if d[1] == True]
                     if valid_only:
                          final_detections_to_draw = valid_only

                for det, face_passed, details in final_detections_to_draw:
                     x1, y1, x2, y2, conf, landmarks = det
                     
                     color = (0, 255, 0) if face_passed else (0, 0, 255)
                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                     
                     label_text = f"Conf: {conf:.2f}"
                     if "label" in details:
                          label_text += details["label"]
                     elif "quality" in details: 
                          label_text += f" Q:{details['quality']:.1f}"
                     
                     cv2.putText(annotated_frame, label_text, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                preview_callback(annotated_frame)
                         
            frame_idx += 1

        cap.release()
        
        # Post-process: filter out short no-face runs
        if keep_no_face and min_no_face_duration > 0 and valid_frames and no_face_frames:
            min_no_face_frames_count = int(min_no_face_duration * fps)
            # Find consecutive runs of no-face frames within valid_frames
            frames_to_remove = set()
            run_start = None
            run_frames = []
            
            for f in valid_frames:
                if f in no_face_frames:
                    if run_start is None:
                        run_start = f
                        run_frames = [f]
                    else:
                        run_frames.append(f)
                else:
                    # End of a no-face run
                    if run_start is not None:
                        if len(run_frames) < min_no_face_frames_count:
                            frames_to_remove.update(run_frames)
                        run_start = None
                        run_frames = []
            
            # Handle trailing run
            if run_start is not None and len(run_frames) < min_no_face_frames_count:
                frames_to_remove.update(run_frames)
            
            if frames_to_remove:
                valid_frames = [f for f in valid_frames if f not in frames_to_remove]
        
        if not valid_frames:
            return [], "No valid frames found matching criteria."

        # Merge frames into clips
        if min_face_quality > 0:
             # Strict Mode: Very low tolerance for gaps aka blurry frames
             gap_tolerance = 1 
        else:
             # Standard Mode: Allow small dropouts
             gap_tolerance = int(fps * 0.5) 
        
        segments = []
        if valid_frames:
            start = valid_frames[0]
            prev = valid_frames[0]
            prev_is_noface = (valid_frames[0] in no_face_frames)
            
            for f in valid_frames[1:]:
                cur_is_noface = (f in no_face_frames)
                # Force a segment break when transitioning between face and no-face regions
                face_type_changed = (cur_is_noface != prev_is_noface)
                
                if f - prev > gap_tolerance or face_type_changed:
                    # Check duration
                    duration = (prev - start + 1) / fps
                    if duration >= min_duration:
                        seg = self._build_segment(start, prev, frame_genders, no_face_frames)
                        segments.append(seg)
                    start = f
                    prev_is_noface = cur_is_noface
                prev = f
            
            # Check last segment
            duration = (prev - start + 1) / fps
            if duration >= min_duration:
                seg = self._build_segment(start, prev, frame_genders, no_face_frames)
                segments.append(seg)

        return segments, "Scanning complete."

    def _build_segment(self, start_frame, end_frame, frame_genders, no_face_frames=None):
        """
        Build a segment dict with metadata including gender tags.
        frame_genders: dict mapping frame_idx -> set of detected genders.
        no_face_frames: set of frame indices where no face was detected.
        """
        # Collect all genders detected across frames in this segment
        segment_genders = set()
        for f in range(start_frame, end_frame + 1):
            if f in frame_genders:
                segment_genders.update(frame_genders[f])
        
        has_male = "Male" in segment_genders
        
        # Determine if this segment contains any face-detected frames
        has_face = True
        if no_face_frames is not None:
            # Check if ALL frames in this segment are no-face frames
            all_noface = all(f in no_face_frames for f in range(start_frame, end_frame + 1))
            has_face = not all_noface
        
        seg = {
            'start_frame': start_frame,
            'end_frame': end_frame,
            'has_male': has_male,
            'has_face': has_face,
            'genders': sorted(list(segment_genders))
        }
        return seg

    def render_video(self, video_path, output_path, segments, progress_callback=None, stop_event=None):
        try:
            import subprocess
            import tempfile
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Calculate total frames to write for progress tracking
            total_frames_to_write = sum(
                seg['end_frame'] - seg['start_frame'] + 1 for seg in segments
            )
            
            # Write video frames with OpenCV for frame-precise cuts
            # Use a temp file for video-only, then mux audio
            temp_dir = os.path.dirname(output_path) or '.'
            temp_video = os.path.join(temp_dir, '_facecut_temp_video.mp4')
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_video, fourcc, fps, (frame_w, frame_h))
            
            if not writer.isOpened():
                cap.release()
                return False, "Failed to create video writer."
            
            frames_written = 0
            
            for seg_idx, seg in enumerate(segments):
                if stop_event and stop_event.is_set():
                    writer.release()
                    cap.release()
                    if os.path.exists(temp_video):
                        os.remove(temp_video)
                    return False, "Rendering stopped by user."
                
                start_frame = seg['start_frame']
                end_frame = seg['end_frame']
                
                # Seek to exact start frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                
                for f_idx in range(start_frame, end_frame + 1):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    writer.write(frame)
                    frames_written += 1
                    
                    if progress_callback and frames_written % 30 == 0:
                        progress_callback(frames_written / total_frames_to_write * 0.7)  # 70% for video
            
            writer.release()
            cap.release()
            
            if frames_written == 0:
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                return False, "No frames written."
            
            # Now mux audio from original video
            # Extract and concatenate audio segments using MoviePy, then mux with ffmpeg
            has_audio = False
            temp_audio = os.path.join(temp_dir, '_facecut_temp_audio.m4a')
            
            try:
                original_clip = VideoFileClip(video_path)
                if original_clip.audio is not None:
                    audio_subclips = []
                    for seg in segments:
                        start_time = seg['start_frame'] / fps
                        end_time = min((seg['end_frame'] + 1) / fps, original_clip.duration)
                        if end_time > start_time:
                            audio_subclips.append(original_clip.audio.subclip(start_time, end_time))
                    
                    if audio_subclips:
                        from moviepy.editor import concatenate_audioclips
                        final_audio = concatenate_audioclips(audio_subclips)
                        final_audio.write_audiofile(temp_audio, codec='aac', verbose=False, logger=None)
                        final_audio.close()
                        has_audio = True
                
                original_clip.close()
            except Exception as e:
                print(f"Audio extraction warning: {e}")
                has_audio = False
            
            if progress_callback:
                progress_callback(0.85)
            
            # Mux video + audio with ffmpeg for best quality
            if has_audio and os.path.exists(temp_audio):
                try:
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video,
                        '-i', temp_audio,
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '18',
                        '-c:a', 'aac',
                        '-b:a', '192k',
                        '-shortest',
                        '-movflags', '+faststart',
                        output_path
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                    
                    if result.returncode != 0:
                        # Fallback: try without audio
                        print(f"ffmpeg mux failed: {result.stderr[:500]}")
                        cmd_noaudio = [
                            'ffmpeg', '-y',
                            '-i', temp_video,
                            '-c:v', 'libx264',
                            '-preset', 'medium',
                            '-crf', '18',
                            '-movflags', '+faststart',
                            output_path
                        ]
                        subprocess.run(cmd_noaudio, capture_output=True, text=True, timeout=600)
                        
                except FileNotFoundError:
                    # ffmpeg not available, use MoviePy to re-encode
                    print("ffmpeg not found, falling back to MoviePy mux...")
                    from moviepy.editor import AudioFileClip
                    video_clip = VideoFileClip(temp_video)
                    audio_clip = AudioFileClip(temp_audio)
                    video_clip = video_clip.set_audio(audio_clip)
                    video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
                    video_clip.close()
                    audio_clip.close()
            else:
                # No audio — just re-encode with ffmpeg or copy
                try:
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', temp_video,
                        '-c:v', 'libx264',
                        '-preset', 'medium',
                        '-crf', '18',
                        '-movflags', '+faststart',
                        output_path
                    ]
                    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                except FileNotFoundError:
                    # No ffmpeg — just rename temp as output
                    shutil.move(temp_video, output_path)
                    temp_video = None  # Already moved
            
            # Cleanup temp files
            if temp_video and os.path.exists(temp_video):
                os.remove(temp_video)
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            if progress_callback:
                progress_callback(1.0)
            
            return True, "Processing complete."
        except Exception as e:
            # Cleanup on error
            temp_video_path = os.path.join(os.path.dirname(output_path) or '.', '_facecut_temp_video.mp4')
            temp_audio_path = os.path.join(os.path.dirname(output_path) or '.', '_facecut_temp_audio.m4a')
            for tmp in [temp_video_path, temp_audio_path]:
                if os.path.exists(tmp):
                    try: os.remove(tmp)
                    except: pass
            return False, f"Video editing error: {str(e)}"

    def process_frame(self, frame, min_conf, max_angle=90, target_gender="All", rec_threshold=0.5, min_face_quality=0.0):
        """
        Process a single frame for preview.
        Returns: annotated_frame, is_valid_frame
        """
        # Reset EMA when starting preview processing (optional, can be removed if continuous smoothing desired)
        # self.quality_ema = None
        
        detections = self.detect_faces(frame, min_conf, max_angle)
        
        annotated_frame = frame.copy()
        frame_valid = False
        img_h, img_w, _ = frame.shape
        
        for x1, y1, x2, y2, conf, landmarks in detections:
            face_valid = True
            
            gender_label = ""
            
            # Combined Gender / Rec / Quality Check
            needs_crop = (target_gender != "All") or (self.reference_embedding is not None) or (min_face_quality > 0)
            
            if face_valid and needs_crop:
                    # Try alignment for Rec/Quality
                    align_img = None
                    if landmarks is not None:
                         align_img = self.align_face(frame, landmarks)
                    
                    crop_img = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
                    
                    if crop_img.size > 0:
                        # Gender (use padded bbox crop matching InsightFace preprocessing)
                        if target_gender != "All":
                             gender = self.predict_gender(frame, (x1, y1, x2, y2))
                             gender_label += f" {gender}"
                             if gender == "Unknown" or gender != target_gender:
                                 face_valid = False
                        
                        # Rec / Quality (use aligned if available, else crop)
                        if self.reference_embedding is not None or min_face_quality > 0:
                            use_img = align_img if align_img is not None else crop_img
                            # Ensure use_img is valid
                            if use_img is None or use_img.size == 0:
                                use_img = crop_img

                            if use_img.size > 0:
                                # Tilt Boost - DISABLE for consistency with strict check
                                tilt_boost = 1.0

                                emb, quality = self.get_embedding(use_img, tilt_boost=tilt_boost)
                                
                                if min_face_quality > 0:
                                    gender_label += f" Q:{quality:.1f}"
                                    if quality < min_face_quality:
                                        face_valid = False
                                
                                if self.reference_embedding is not None:
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

    def process_frame_smart(self, frame, min_conf, max_angle=90, target_gender="All", rec_threshold=0.5, min_face_quality=0.0):
        """
        Process a single frame for preview (Smart Rotation enabled).
        Returns: annotated_frame, is_valid_frame
        """
        # Convert to RGB once for all operations
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detect_faces(frame, min_conf, max_angle)
        
        annotated_frame = frame.copy()
        frame_valid = False
        img_h, img_w, _ = frame.shape
        
        low_qual_candidates = 0
        evaluated_detections = []
        is_valid = False

        # Phase 1: Standard Evaluation
        for det in detections:
             passed, low_qual_fail, details = self.evaluate_face(frame, det, target_gender, rec_threshold, min_face_quality, rgb_frame=rgb_frame)
             if passed:
                  is_valid = True
             if low_qual_fail:
                  low_qual_candidates += 1
             
             evaluated_detections.append((det, passed, details))
             
        # Phase 2: Smart Rotation Retry
        if not is_valid and low_qual_candidates > 0 and max_angle > 60 and min_face_quality > 0:
             # Retry with rotations
             h, w = frame.shape[:2]
             rotations = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
             
             for rot in rotations:
                  frame_rot = cv2.rotate(frame, rot)
                  rgb_rot = cv2.rotate(rgb_frame, rot)
                  dets_rot = self._detect_single_pass(frame_rot, rgb_rot, min_conf, max_angle)
                  
                  for d_rot in dets_rot:
                       # Map back
                       x1, y1, x2, y2, conf, lms = d_rot
                       (rx1, ry1, rx2, ry2), rlms = self.rotate_coords((x1, y1, x2, y2), lms, rot, w, h)
                       det_orig = (rx1, ry1, rx2, ry2, conf, rlms)
                       
                       # Evaluate on ROTATED FRAME for best quality
                       passed, low_qual_fail, details = self.evaluate_face(frame_rot, d_rot, target_gender, rec_threshold, min_face_quality, force_tilt_boost=True, rgb_frame=rgb_rot)
                       
                       if passed:
                            is_valid = True
                            evaluated_detections.append((det_orig, passed, details))
                            break
                  
                  if is_valid:
                       break
        
        frame_valid = is_valid

        # If we succeeded in smart rotation, hide failed detections.
        final_detections_to_draw = evaluated_detections
        if is_valid and low_qual_candidates > 0:
             # Keep only valid ones
             valid_only = [d for d in evaluated_detections if d[1] == True]
             if valid_only:
                  final_detections_to_draw = valid_only

        for det, face_passed, details in final_detections_to_draw:
            x1, y1, x2, y2, conf, landmarks = det
            
            color = (0, 255, 0) if face_passed else (0, 0, 255)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"Conf: {conf:.2f}"
            if "label" in details:
                 label_text += details["label"]
            
            cv2.putText(annotated_frame, label_text, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        return annotated_frame, frame_valid
