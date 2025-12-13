import streamlit as st
import tempfile
import os
from processor import VideoProcessor

st.set_page_config(page_title="FaceCut", page_icon="✂️")

st.title("✂️ FaceCut: Smart Video Trimmer")
st.markdown("Upload a video and automatically cut it to keep only the parts where a face is visible and looking at the camera.")

# Sidebar Configuration
st.sidebar.header("Configuration")
min_confidence = st.sidebar.slider("Detection Sensitivity (Confidence)", 0.0, 1.0, 0.6, 0.05, help="Higher values mean strictly clearer faces.")
require_unobstructed = st.sidebar.checkbox("Strict: No Obstruction", value=False, help="If checked, frames where the face is partially out of view will be discarded.")
obstruction_margin = st.sidebar.slider("Edge Margin (%)", 0, 25, 0, 1, help="Percentage of frame edge to consider as obstructed.") / 100.0
min_duration = st.sidebar.slider("Min Segment Duration (s)", 0.0, 2.0, 0.5, 0.1, help="Discard segments shorter than this.")


uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov', 'avi'])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    st.video(video_path)
    
    if st.button("Process Video"):
        processor = VideoProcessor()
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        output_path = os.path.join(tempfile.gettempdir(), "output_facecut.mp4")
        
        def update_progress(p):
            progress_bar.progress(p)
            status_text.text(f"Processing... {int(p*100)}%")

        try:
            success, message = processor.process_video(
                video_path, 
                output_path, 
                min_confidence, 
                require_unobstructed,
                obstruction_margin=obstruction_margin,
                min_duration=min_duration,
                progress_callback=update_progress
            )
            
            if success:
                status_text.text("Processing Complete!")
                progress_bar.progress(1.0)
                st.success("Video processed successfully!")
                st.video(output_path)
                
                with open(output_path, "rb") as f:
                    st.download_button("Download Processed Video", f, file_name="facecut_output.mp4")
            else:
                st.error(f"Error: {message}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Cleanup input temp file
            try:
                os.remove(video_path)
            except:
                pass
