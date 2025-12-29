import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import os
import cv2
import json
from PIL import Image, ImageTk
from processor import VideoProcessor

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class FaceCutApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("FaceCut - YOLOv11 Strict Edition")
        self.geometry("1300x700") # Increased width for 3 columns

        # Grid configuration: 0=Sidebar, 1=ClipList, 2=Preview
        self.grid_columnconfigure(0, weight=0, minsize=250)
        self.grid_columnconfigure(1, weight=0, minsize=300)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Left Sidebar (Controls) ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(20, weight=1)

        self.label_title = ctk.CTkLabel(self.sidebar, text="FaceCut AI", font=ctk.CTkFont(size=20, weight="bold"))
        self.label_title.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_select = ctk.CTkButton(self.sidebar, text="Select Video", command=self.select_file)
        self.btn_select.grid(row=1, column=0, padx=20, pady=10)
        
        self.lbl_file = ctk.CTkLabel(self.sidebar, text="No file selected", wraplength=200)
        self.lbl_file.grid(row=2, column=0, padx=20, pady=(0, 10))

        # Model Selection
        self.lbl_model = ctk.CTkLabel(self.sidebar, text="Detection Model")
        self.lbl_model.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.model_var = ctk.StringVar(value="yolo")
        self.radio_s3fd = ctk.CTkRadioButton(self.sidebar, text="S3FD (Accurate)", variable=self.model_var, value="s3fd", command=self.change_model)
        self.radio_s3fd.grid(row=4, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.radio_yolo = ctk.CTkRadioButton(self.sidebar, text="YOLOv11 (Fast)", variable=self.model_var, value="yolo", command=self.change_model)
        self.radio_yolo.grid(row=5, column=0, padx=20, pady=(5, 5), sticky="w")
        
        self.radio_mp = ctk.CTkRadioButton(self.sidebar, text="MediaPipe (Balanced)", variable=self.model_var, value="mediapipe", command=self.change_model)
        self.radio_mp.grid(row=6, column=0, padx=20, pady=(5, 10), sticky="w")

        # Settings
        self.lbl_conf = ctk.CTkLabel(self.sidebar, text="Confidence: 0.50")
        self.lbl_conf.grid(row=7, column=0, padx=20, pady=(10, 0), sticky="w")
        self.slider_conf = ctk.CTkSlider(self.sidebar, from_=0.1, to=1.0, number_of_steps=90, command=self.update_conf_label)
        self.slider_conf.set(0.6)
        self.slider_conf.grid(row=8, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # Margin Slider
        self.lbl_margin = ctk.CTkLabel(self.sidebar, text="Edge Margin: 0%")
        self.lbl_margin.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.slider_margin = ctk.CTkSlider(self.sidebar, from_=0, to=40, number_of_steps=40, command=self.update_margin_label)
        self.slider_margin.set(0)
        self.slider_margin.grid(row=10, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Obstruction Threshold
        self.lbl_obstruction = ctk.CTkLabel(self.sidebar, text="Max Obstruction: 0% (Strict)")
        self.lbl_obstruction.grid(row=11, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.slider_obstruction = ctk.CTkSlider(self.sidebar, from_=0.0, to=0.5, number_of_steps=50, command=self.update_obstruction_label)
        self.slider_obstruction.set(0.0)
        self.slider_obstruction.grid(row=12, column=0, padx=20, pady=(0, 10), sticky="ew")
        
        # Min Duration Slider
        self.lbl_duration = ctk.CTkLabel(self.sidebar, text="Min Duration: 0.5s")
        self.lbl_duration.grid(row=13, column=0, padx=20, pady=(10, 0), sticky="w")
        self.slider_duration = ctk.CTkSlider(self.sidebar, from_=0.0, to=2.0, number_of_steps=20, command=self.update_duration_label)
        self.slider_duration.set(0.5)
        self.slider_duration.grid(row=14, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Max Angle Slider (MediaPipe only)
        self.lbl_angle = ctk.CTkLabel(self.sidebar, text="Max Angle: 45°")
        self.lbl_angle.grid(row=15, column=0, padx=20, pady=(10, 0), sticky="w")
        self.slider_angle = ctk.CTkSlider(self.sidebar, from_=10, to=90, number_of_steps=80, command=self.update_angle_label)
        self.slider_angle.set(45)
        self.slider_angle.grid(row=16, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Gender Selection
        self.lbl_gender = ctk.CTkLabel(self.sidebar, text="Keep Gender")
        self.lbl_gender.grid(row=17, column=0, padx=20, pady=(10, 0), sticky="w")
        self.combo_gender = ctk.CTkComboBox(self.sidebar, values=["All", "Male", "Female"])
        self.combo_gender.set("All")
        self.combo_gender.grid(row=18, column=0, padx=20, pady=(0, 10), sticky="ew")

        # Face Recognition
        self.lbl_rec = ctk.CTkLabel(self.sidebar, text="Face Recognition")
        self.lbl_rec.grid(row=19, column=0, padx=20, pady=(10, 0), sticky="w")
        
        self.btn_ref_face = ctk.CTkButton(self.sidebar, text="Set Ref Face", command=self.set_reference_face)
        self.btn_ref_face.grid(row=20, column=0, padx=20, pady=(5, 5), sticky="ew")
        
        self.lbl_ref_status = ctk.CTkLabel(self.sidebar, text="No Ref Face", font=ctk.CTkFont(size=10))
        self.lbl_ref_status.grid(row=21, column=0, padx=20, pady=(0, 5))
        
        self.lbl_sim = ctk.CTkLabel(self.sidebar, text="Sim Threshold: 0.50")
        self.lbl_sim.grid(row=22, column=0, padx=20, pady=(5, 0), sticky="w")
        
        self.slider_sim = ctk.CTkSlider(self.sidebar, from_=0.1, to=1.0, number_of_steps=90, command=self.update_sim_label)
        self.slider_sim.set(0.5)
        self.slider_sim.grid(row=23, column=0, padx=20, pady=(0, 10), sticky="ew")

        self.btn_process = ctk.CTkButton(self.sidebar, text="Start Processing", command=self.start_processing, state="disabled", fg_color="green")
        self.btn_process.grid(row=24, column=0, padx=20, pady=(20, 10))

        self.btn_stop = ctk.CTkButton(self.sidebar, text="Stop", command=self.stop_processing, state="disabled", fg_color="red")
        self.btn_stop.grid(row=25, column=0, padx=20, pady=(0, 10))

        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.grid(row=26, column=0, padx=20, pady=10, sticky="ew")
        self.progress_bar.set(0)

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Ready")
        self.lbl_status.grid(row=27, column=0, padx=20, pady=10)


        # --- Middle Column (Clip List) ---
        self.clip_frame = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.clip_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=0)
        self.clip_frame.grid_rowconfigure(1, weight=1)
        self.clip_frame.grid_columnconfigure(0, weight=1)

        self.lbl_clips_title = ctk.CTkLabel(self.clip_frame, text="Clips Found: 0", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_clips_title.grid(row=0, column=0, padx=10, pady=10)

        self.scroll_clips = ctk.CTkScrollableFrame(self.clip_frame)
        self.scroll_clips.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.clip_actions_frame = ctk.CTkFrame(self.clip_frame)
        self.clip_actions_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=10)
        
        self.btn_toggle_all = ctk.CTkButton(self.clip_actions_frame, text="Toggle All", command=self.toggle_all_clips, width=100)
        self.btn_toggle_all.pack(side="left", padx=5)

        self.btn_merge_clips = ctk.CTkButton(self.clip_actions_frame, text="Merge", command=self.merge_selected_clips, width=100, fg_color="green")
        self.btn_merge_clips.pack(side="right", padx=5)


        # --- Right Main Area (Preview) ---
        self.preview_frame = ctk.CTkFrame(self)
        self.preview_frame.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")
        self.preview_frame.grid_rowconfigure(0, weight=1)
        self.preview_frame.grid_columnconfigure(0, weight=1)

        self.lbl_video = ctk.CTkLabel(self.preview_frame, text="Video Preview", corner_radius=10)
        self.lbl_video.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Playback Controls
        self.controls_frame = ctk.CTkFrame(self.preview_frame, height=50)
        self.controls_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        self.btn_play = ctk.CTkButton(self.controls_frame, text="Play", width=80, command=self.toggle_play, state="disabled")
        self.btn_play.pack(side="left", padx=10, pady=10)

        self.slider_seek = ctk.CTkSlider(self.controls_frame, from_=0, to=100, command=self.seek_video)
        self.slider_seek.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        self.slider_seek.set(0)

        # State
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.processor = None
        self.total_frames = 0
        self.current_frame_idx = 0
        self.fps = 30
        self.update_job = None
        self.stop_event = threading.Event()
        self.stop_frame_idx = None # For previewing segments
        
        self.detected_segments = []
        self.clip_check_vars = []

        # Initialize processor in background
        threading.Thread(target=self.init_processor, daemon=True).start()

    def init_processor(self):
        self.lbl_status.configure(text="Loading AI Model...")
        # Default to S3FD if variable not set yet (though it is set in init)
        model = self.model_var.get() if hasattr(self, 'model_var') else 'yolo'
        self.processor = VideoProcessor(model_type=model)
        self.lbl_status.configure(text=f"AI Ready ({model.upper()})")

    def change_model(self):
        # When radio button changes, re-init processor
        # We do this in a thread to not freeze UI
        model = self.model_var.get()
        if model == 'mediapipe':
            self.lbl_angle.grid()
            self.slider_angle.grid()
        else:
            self.lbl_angle.grid_remove()
            self.slider_angle.grid_remove()
            
        threading.Thread(target=self.init_processor, daemon=True).start()

    def update_conf_label(self, value):
        self.lbl_conf.configure(text=f"Confidence: {value:.2f}")

    def update_margin_label(self, value):
        self.lbl_margin.configure(text=f"Edge Margin: {int(value)}%")

    def update_obstruction_label(self, value):
        percent = int(value * 100)
        text = f"Max Obstruction: {percent}%"
        if percent == 0:
            text += " (Strict)"
        self.lbl_obstruction.configure(text=text)

    def update_duration_label(self, value):
        self.lbl_duration.configure(text=f"Min Duration: {value:.1f}s")

    def update_angle_label(self, value):
        self.lbl_angle.configure(text=f"Max Angle: {int(value)}°")

    def update_sim_label(self, value):
        self.lbl_sim.configure(text=f"Sim Threshold: {value:.2f}")

    def set_reference_face(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if file_path:
            success, msg = self.processor.set_reference_face(file_path)
            if success:
                self.lbl_ref_status.configure(text=os.path.basename(file_path), text_color="green")
                messagebox.showinfo("Success", "Reference face set successfully.")
            else:
                self.lbl_ref_status.configure(text="Error", text_color="red")
                messagebox.showerror("Error", msg)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.mov *.avi")])
        if file_path:
            self.video_path = file_path
            self.lbl_file.configure(text=os.path.basename(file_path))
            self.btn_process.configure(state="normal")
            self.load_video(file_path)
            self.load_segments()

    def load_video(self, path):
        if self.cap:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0
        self.stop_frame_idx = None
        self.slider_seek.configure(to=self.total_frames)
        self.slider_seek.set(0)
        
        self.btn_play.configure(state="normal")
        self.show_frame()

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.btn_play.configure(text="Play")
            self.stop_frame_idx = None # Reset stop if manually paused
            if self.update_job:
                self.after_cancel(self.update_job)
                self.update_job = None
        else:
            self.is_playing = True
            self.btn_play.configure(text="Pause")
            self.play_video()

    def play_video(self):
        if not self.is_playing or not self.cap:
            return

        if self.current_frame_idx >= self.total_frames:
            self.is_playing = False
            self.btn_play.configure(text="Play")
            return

        # Check if we reached the segment end
        if self.stop_frame_idx is not None and self.current_frame_idx >= self.stop_frame_idx:
             self.is_playing = False
             self.btn_play.configure(text="Play")
             self.stop_frame_idx = None
             return

        self.show_frame()
        self.current_frame_idx += 1
        self.slider_seek.set(self.current_frame_idx)
        
        delay = int(1000 / self.fps)
        self.update_job = self.after(delay, self.play_video)

    def preview_segment(self, start, end):
        self.stop_frame_idx = end
        self.seek_video(start)
        if not self.is_playing:
            self.toggle_play()

    def seek_video(self, value):
        if not self.cap:
            return
        self.current_frame_idx = int(value)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        self.show_frame()

    def show_frame(self):
        if not self.cap:
            return
            
        if not self.is_playing:
             self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)

        ret, frame = self.cap.read()
        if ret:
            h, w, _ = frame.shape
            display_h = 480
            scale = display_h / h
            display_w = int(w * scale)
            
            if frame is not None:
                min_conf = self.slider_conf.get()
                # Use obstruction slider
                obstruction_thresh = self.slider_obstruction.get()
                margin = self.slider_margin.get() / 100.0
                max_angle = self.slider_angle.get()
                target_gender = self.combo_gender.get()
                rec_thresh = self.slider_sim.get()
                frame, is_valid = self.processor.process_frame(frame, min_conf, obstruction_thresh, margin, max_angle, target_gender, rec_thresh)
            
            frame = cv2.resize(frame, (display_w, display_h))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.lbl_video.configure(image=imgtk, text="")
            self.lbl_video.image = imgtk

    def start_processing(self):
        if not self.video_path:
            return

        self.btn_process.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.lbl_status.configure(text="Initializing...")
        self.progress_bar.set(0)
        
        if self.is_playing:
            self.toggle_play()

        min_conf = self.slider_conf.get()
        obstruction_thresh = self.slider_obstruction.get()
        margin = self.slider_margin.get() / 100.0
        min_dur = self.slider_duration.get()
        max_angle = self.slider_angle.get()
        target_gender = self.combo_gender.get()
        rec_thresh = self.slider_sim.get()
        self.stop_event.clear()

        # Clear existing clips
        self.clear_clip_list()

        threading.Thread(target=self.run_processing, args=(min_conf, obstruction_thresh, margin, min_dur, max_angle, target_gender, rec_thresh)).start()

    def stop_processing(self):
        self.stop_event.set()
        self.lbl_status.configure(text="Stopping...")
        self.btn_stop.configure(state="disabled")

    def clear_clip_list(self):
        for widget in self.scroll_clips.winfo_children():
            widget.destroy()
        self.clip_check_vars = []
        self.detected_segments = []
        self.lbl_clips_title.configure(text="Clips Found: 0")

    def populate_clip_list(self, segments):
        self.detected_segments = segments
        self.lbl_clips_title.configure(text=f"Clips Found: {len(segments)}")
        
        for i, seg in enumerate(segments):
            start = seg['start_frame']
            end = seg['end_frame']
            duration = (end - start + 1) / self.fps
            start_time = start / self.fps
            
            chk_var = ctk.BooleanVar(value=True)
            self.clip_check_vars.append(chk_var)
            
            # Row frame
            row = ctk.CTkFrame(self.scroll_clips, fg_color="transparent")
            row.pack(fill="x", pady=2)
            
            chk = ctk.CTkCheckBox(row, text=f"Clip {i+1}: {duration:.1f}s (at {start_time:.1f}s)", variable=chk_var)
            chk.pack(side="left")
            
            # Preview button
            btn_prev = ctk.CTkButton(row, text="▶", width=30, height=20, 
                                     command=lambda s=start, e=end: self.preview_segment(s, e))
            btn_prev.pack(side="right", padx=5)

    def toggle_all_clips(self):
        # Check if all are true
        if not self.clip_check_vars:
            return
        all_checked = all(v.get() for v in self.clip_check_vars)
        for v in self.clip_check_vars:
            v.set(not all_checked)

    def merge_selected_clips(self):
        selected_segments = []
        for i, var in enumerate(self.clip_check_vars):
            if var.get():
                selected_segments.append(self.detected_segments[i])
        
        if not selected_segments:
            messagebox.showwarning("Warning", "No clips selected!")
            return
            
        self.start_rendering(selected_segments)


    def run_processing(self, min_conf, obstruction_thresh, margin, min_dur, max_angle, target_gender, rec_thresh):
        try:
            # Phase 1: Scan
            def update_prog(p):
                # Only go up by 50% for scanning? Or 100% then reset?
                # Let's say scanning is the "Start Processing" part.
                self.progress_bar.set(p)
                self.lbl_status.configure(text=f"Scanning... {int(p*100)}%")

            def update_preview(frame):
                # Resize and display frame
                h, w, _ = frame.shape
                display_h = 480
                scale = display_h / h
                display_w = int(w * scale)
                
                frame = cv2.resize(frame, (display_w, display_h))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Use after to update UI from thread
                self.after(0, lambda: self.lbl_video.configure(image=imgtk, text=""))
                self.after(0, lambda: setattr(self.lbl_video, 'image', imgtk))

            segments, msg = self.processor.scan_video(
                self.video_path,
                min_conf,
                obstruction_threshold=obstruction_thresh,
                obstruction_margin=margin,
                min_duration=min_dur,
                max_angle=max_angle,
                target_gender=target_gender,
                rec_threshold=rec_thresh,
                progress_callback=update_prog,
                preview_callback=update_preview,
                stop_event=self.stop_event
            )

            if segments is None: # Stopped
                 self.lbl_status.configure(text="Stopped")
                 self.btn_process.configure(state="normal")
                 self.btn_select.configure(state="normal")
                 self.btn_stop.configure(state="disabled")
                 return
            
            if not segments:
                self.lbl_status.configure(text="No clips found")
                messagebox.showinfo("Info", msg)
                self.btn_process.configure(state="normal")
                self.btn_select.configure(state="normal")
                self.btn_stop.configure(state="disabled")
                return

            self.lbl_status.configure(text=f"Found {len(segments)} clips")
            
            # Populate Clip List on main thread
            self.after(0, lambda: self.populate_clip_list(segments))
            self.after(0, lambda: self.save_segments(segments))
            self.after(0, lambda: self.btn_process.configure(state="normal"))
            self.after(0, lambda: self.btn_select.configure(state="normal"))
            self.after(0, lambda: self.btn_stop.configure(state="disabled"))


        except Exception as e:
            self.lbl_status.configure(text="Error")
            messagebox.showerror("Error", str(e))
            self.btn_process.configure(state="normal")
            self.btn_select.configure(state="normal")
            self.btn_stop.configure(state="disabled")

    def start_rendering(self, selected_segments):
        self.btn_process.configure(state="disabled")
        self.btn_select.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.stop_event.clear()
        
        threading.Thread(target=self.run_rendering, args=(selected_segments,)).start()

    def run_rendering(self, segments):
        try:
            output_path = os.path.splitext(self.video_path)[0] + "_facecut.mp4"
            
            def update_prog(p):
                self.progress_bar.set(p)
                self.lbl_status.configure(text=f"Rendering... {int(p*100)}%")
            
            success, msg = self.processor.render_video(
                self.video_path,
                output_path,
                segments,
                progress_callback=update_prog,
                stop_event=self.stop_event
            )
            
            if success:
                self.lbl_status.configure(text="Done!")
                messagebox.showinfo("Success", f"Video saved to:\n{output_path}")
            else:
                 self.lbl_status.configure(text="Failed" if "stopped" not in msg else "Stopped")
                 if "stopped" not in msg:
                    messagebox.showerror("Error", msg)

        except Exception as e:
            self.lbl_status.configure(text="Error")
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_process.configure(state="normal")
            self.btn_select.configure(state="normal")
            self.btn_stop.configure(state="disabled")

    def save_segments(self, segments):
        if not self.video_path:
            return
            
        try:
            json_path = os.path.splitext(self.video_path)[0] + "_segments.json"
            data = {
                "video_path": self.video_path,
                "segments": segments,
                "parameters": {
                    "min_conf": self.slider_conf.get(),
                    "margin": self.slider_margin.get(),
                    "min_duration": self.slider_duration.get(),
                    "max_angle": self.slider_angle.get(),
                    "obstruction_threshold": self.slider_obstruction.get(),
                    "target_gender": self.combo_gender.get(),
                    "rec_threshold": self.slider_sim.get() 
                }
            }
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Segments saved to {json_path}")
        except Exception as e:
            print(f"Failed to save segments: {e}")

    def load_segments(self):
        self.clear_clip_list()
        if not self.video_path:
            return

        json_path = os.path.splitext(self.video_path)[0] + "_segments.json"
        if not os.path.exists(json_path):
            return

        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            if "segments" in data:
                print(f"Loaded {len(data['segments'])} segments from {json_path}")
                self.populate_clip_list(data['segments'])
                
                # Load parameters
                if "parameters" in data:
                    params = data["parameters"]
                    if "min_conf" in params:
                        self.slider_conf.set(params["min_conf"])
                        self.update_conf_label(params["min_conf"])
                    if "margin" in params:
                        self.slider_margin.set(params["margin"])
                        self.update_margin_label(params["margin"])
                    if "min_duration" in params:
                        self.slider_duration.set(params["min_duration"])
                        self.update_duration_label(params["min_duration"])
                    if "max_angle" in params:
                        self.slider_angle.set(params["max_angle"])
                        self.update_angle_label(params["max_angle"])
                    
                    # Handle old obstruction_check
                    if "obstruction_check" in params:
                        # If old bool was True (strict), set thresh to 0.0. If False, maybe 0.5?
                        if params["obstruction_check"]:
                            self.slider_obstruction.set(0.0)
                            self.update_obstruction_label(0.0)
                        else:
                            # Lax
                            self.slider_obstruction.set(0.5)
                            self.update_obstruction_label(0.5)

                    if "obstruction_threshold" in params:
                        val = params["obstruction_threshold"]
                        self.slider_obstruction.set(val)
                        self.update_obstruction_label(val)

                    if "target_gender" in params:
                        self.combo_gender.set(params["target_gender"])
                    if "rec_threshold" in params:
                        self.slider_sim.set(params["rec_threshold"])
                        self.update_sim_label(params["rec_threshold"])

                self.lbl_status.configure(text=f"Loaded {len(data['segments'])} saved clips")
        except Exception as e:
            print(f"Failed to load segments: {e}")

if __name__ == "__main__":
    app = FaceCutApp()
    app.mainloop()
