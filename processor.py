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
        
        # FAN (Face Alignment Network) for Strict Validation
        self.fan = None
        
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


    def ensure_fan(self):
        if self.fan is None:
            print("Loading FAN (Face Alignment Network) for strict validation...")
            try:
                # Use CPU/Cuda based on avail. '2D' landmarks.
                self.fan = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, 
                                                      device=self.device, 
                                                      face_detector='blazeface') # Use lightweight internal detector?
                                                      # Actually we pass the rect, so detector matters less, but blazeface is standard.
                print("FAN loaded.")
            except Exception as e:
                print(f"Failed to load FAN: {e}")
                
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





    def get_embedding(self, face_img, landmarks=None, tilt_boost=1.0):
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
             blob_input = face_img
             
        blob = cv2.dnn.blobFromImage(blob_input, 1.0/127.5, (112, 112), 
                                   (127.5, 127.5, 127.5), 
                                   swapRB=True, crop=False)
        self.rec_net.setInput(blob)
        embedding = self.rec_net.forward()
        # Quality Score (Feature Norm)
        feature_norm = np.linalg.norm(embedding)
        
        # Composite Quality Score (FAN ONLY Mode)
        # We rely solely on FAN validation (done in evaluate_face) + Raw Feature Norm.
        # No penalties for blur, brightness, or padding.
        composite_score = feature_norm
        
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
                self.reference_embedding, _ = self.get_embedding(face_img)
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
        """
        # If max_angle is high (e.g. > 60), assuming user wants to detect highly tilted faces.
        # We run detection on 0, +90, -90.
        # Otherwise just 0.
        
        # Pass 1: Original
        detections = self._detect_single_pass(frame, min_conf, max_angle)
        
        if max_angle > 60 and len(detections) == 0:
             h, w = frame.shape[:2]
             
             # Pass 2: 90 CW
             frame_90 = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
             # Note: For rotated frame, "upright" face is actually tilted 90 in original.
             # The filter inside _detect_single_pass filters by Yaw/Pitch.
             # In rotated frame, Yaw/Pitch are naturally relative to the face's new "upright" orientation.
             # So a 90-deg tilted face in Original is 0-deg in Rotated.
             # _detect_single_pass WILL accept it (small yaw/pitch).
             dets_90 = self._detect_single_pass(frame_90, min_conf, max_angle)
             
             for det in dets_90:
                  x1, y1, x2, y2, conf, lms = det
                  (rx1, ry1, rx2, ry2), rlms = self.rotate_coords((x1, y1, x2, y2), lms, cv2.ROTATE_90_CLOCKWISE, w, h)
                  detections.append((rx1, ry1, rx2, ry2, conf, rlms))
                  
             # Pass 3: 90 CCW
             frame_n90 = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
             dets_n90 = self._detect_single_pass(frame_n90, min_conf, max_angle)
             
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

    def _detect_single_pass(self, frame, min_conf, max_angle=90):
        """
        Unified detection method.
        Returns list of (x1, y1, x2, y2, conf, landmarks)
        landmarks: list of 5 (x,y) tuples or None.
        """
        detections = []
        
        if self.model_type == 's3fd':
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                        
                        # Pose Filter for YOLO
                        if lms is not None:
                             yaw, pitch, roll = self.estimate_pose_from_landmarks(lms)
                             # Filter Yaw/Pitch but ALLOW Roll (User wants up to 90 deg tilt)
                             if abs(yaw) > max_angle: 
                                  continue 
                             # Pitch check (optional, usually less critical, but good to filter extreme looking up/down)
                             if abs(pitch) > max_angle:
                                  continue
                        
                        detections.append((x1, y1, x2, y2, conf, lms))
        
        elif self.model_type == 'mediapipe':
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
                        
                        # Check Angle with new estimator
                        yaw, pitch, roll = self.estimate_pose_from_landmarks(lms)
                        if abs(yaw) > max_angle or abs(pitch) > max_angle:
                             continue
                        
                        detections.append((x1, y1, x2, y2, conf, lms))
                        
        return detections



    def evaluate_face(self, frame, detection, target_gender, rec_threshold, min_face_quality, force_tilt_boost=False):
        """
        Evaluate a single face against criteria.
        Returns: (passed_filters, low_quality_fail_only, details_dict)
        low_quality_fail_only: True if face failed ONLY due to quality score.
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
                  # Convert to RGB only once if needed? 
                  # Actually detected_faces arg in get_landmarks makes it efficient.
                  rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                  
                  preds = self.fan.get_landmarks(rgb, detected_faces=[(x1, y1, x2, y2)])
                  
                  if preds is None or len(preds) == 0:
                       # FAN Failed -> Bad Face Structure
                       return False, False, {"label": " NoStruct", "quality": 0.0}
                  
                  # FAN Succeeded -> Use refined landmarks
                  landmarks = preds[0] # 68 points
                  
             except Exception as e:
                  print(f"FAN Error: {e}")
                  # Fallback or strict fail? Let's strict fail to be safe.
                  return False, False, {"label": " FanErr", "quality": 0.0}
        
        passed = True
        low_quality_fail = False
        gender_label = ""
        quality_score = 0.0
        rec_score = 0.0
        
        # Gender
        if target_gender != "All":
             face_crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
             if face_crop.size > 0:
                  gender = self.predict_gender(face_crop)
                  gender_label += f" {gender}"
                  if gender != target_gender:
                       passed = False
             else:
                  passed = False
                  
        if not passed:
             return False, False, {"label": gender_label, "quality": 0.0}

        # Rec / Quality
        if min_face_quality > 0 or self.reference_embedding is not None:
             # Try alignment
             align_img = None
             if landmarks is not None:
                  align_img = self.align_face(frame, landmarks)
             
             face_crop = frame[max(0, y1):min(img_h, y2), max(0, x1):min(img_w, x2)]
             use_img = align_img if align_img is not None else face_crop
             
             if use_img is None or use_img.size == 0:
                 use_img = face_crop
                 
             if use_img.size > 0:
                 is_high_conf = (conf > 0.8)
                 
                 # Determine if we need to compute embedding
                 # Case 1: Recognition is enabled (reference_embedding set) -> ALWAYS need embedding
                 # Case 2: Recognition disabled, but Quality check needed AND NOT High Conf -> Need embedding
                 need_embedding = (self.reference_embedding is not None) or (min_face_quality > 0 and not is_high_conf)
                 
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
                         if is_high_conf:
                             # High Conf -> Pass automatically, just show score
                             gender_label += f" Q:{quality:.1f}*"
                         else:
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
                     # Skipped embedding calculation (High Conf + No Rec)
                     if is_high_conf:
                         gender_label += " Q:HighConf"
                         # passed stays True
                 
             else:
                 passed = False
        
        return passed, low_quality_fail, {"label": gender_label, "quality": quality_score}

    def scan_video(self, video_path, min_conf, min_duration=0.0, max_angle=90, target_gender="All", rec_threshold=0.5, min_face_quality=0.0, progress_callback=None, preview_callback=None, stop_event=None):
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
            
            low_qual_candidates = 0
            
            # Phase 1: Standard Evaluation
            # We need to reconstruct the logic using evaluate_face
            # But wait, evaluate_face recalculates everything. 
            # We can just iterate detections.
            
            evaluated_detections = [] # Store (det, details, passed) for preview
            
            for det in detections:
                 passed, low_qual_fail, details = self.evaluate_face(frame, det, target_gender, rec_threshold, min_face_quality)
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
                      dets_rot = self._detect_single_pass(frame_rot, min_conf, max_angle)
                      
                      for d_rot in dets_rot:
                           # Map back
                           x1, y1, x2, y2, conf, lms = d_rot
                           (rx1, ry1, rx2, ry2), rlms = self.rotate_coords((x1, y1, x2, y2), lms, rot, w, h)
                           det_orig = (rx1, ry1, rx2, ry2, conf, rlms)
                           
                           # Evaluate on the ROTATED frame (where face is upright)
                           passed, low_qual_fail, details = self.evaluate_face(frame_rot, d_rot, target_gender, rec_threshold, min_face_quality, force_tilt_boost=True)
                           
                           if passed:
                                is_valid = True
                                evaluated_detections.append((det_orig, passed, details))
                                detections.append(det_orig)
                                break
                      
                      if is_valid:
                           break
            
            if is_valid:
                valid_frames.append(frame_idx)

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

    def process_frame(self, frame, min_conf, max_angle=90, target_gender="All", rec_threshold=0.5, min_face_quality=0.0):
        """
        Process a single frame for preview.
        Returns: annotated_frame, is_valid_frame
        """
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
                        # Gender (use crop, fast)
                        if target_gender != "All":
                             gender = self.predict_gender(crop_img)
                             gender_label += f" {gender}"
                             if gender != target_gender:
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
        detections = self.detect_faces(frame, min_conf, max_angle)
        
        annotated_frame = frame.copy()
        frame_valid = False
        img_h, img_w, _ = frame.shape
        
        low_qual_candidates = 0
        evaluated_detections = []
        is_valid = False

        # Phase 1: Standard Evaluation
        for det in detections:
             passed, low_qual_fail, details = self.evaluate_face(frame, det, target_gender, rec_threshold, min_face_quality)
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
                  dets_rot = self._detect_single_pass(frame_rot, min_conf, max_angle)
                  
                  for d_rot in dets_rot:
                       # Map back
                       x1, y1, x2, y2, conf, lms = d_rot
                       (rx1, ry1, rx2, ry2), rlms = self.rotate_coords((x1, y1, x2, y2), lms, rot, w, h)
                       det_orig = (rx1, ry1, rx2, ry2, conf, rlms)
                       
                       # Evaluate on ROTATED FRAME for best quality
                       passed, low_qual_fail, details = self.evaluate_face(frame_rot, d_rot, target_gender, rec_threshold, min_face_quality, force_tilt_boost=True)
                       
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
