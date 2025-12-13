# FaceCut

FaceCut is a powerful desktop application powered by AI (YOLOv11 & MediaPipe) designed to automatically detect and extract video segments containing faces. It features advanced filtering capabilities such as face verification, angle detection, and obstruction checks to ensure only high-quality clips are retained.

## Features

- **AI-Powered Detection**: Supports **YOLOv11** (Face & Pose) and **MediaPipe** for accurate face detection.
- **Advanced Filtering**:
  - **Confidence Threshold**: Adjust detection sensitivity.
  - **Face Angle Filter**: discard faces based on yaw/pitch (e.g., keep only frontal faces).
  - **Obstruction Detection**: Automatically ignore faces that are partially out of frame.
  - **Minimum Duration**: Filter out clips that are too short.
- **Smart Clip Management**:
  - **Preview Segment**: Instantly preview any detected clip within the app.
  - **Save/Load**: Save your session (detected segments) to JSON and resume later.
  - **Merge & Export**: Select specific clips and merge them into a single video file.
- **User-Friendly GUI**: Built with `customtkinter` for a modern dark-mode interface.

## Requirements

- Windows (Recommended for BAT script usage)
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/) (Required by `moviepy` for video processing)
- CUDA-capable GPU (Optional, but highly recommended for YOLOv11 performance)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/facecut.git
    cd facecut
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

   *Note: Ensure you have PyTorch installed with CUDA support if you have a compatible GPU.*

## Usage

1.  **Start the Application**:
    Run the provided batch file (Windows):
    ```bash
    run_facecut.bat
    ```
    Or run via Python directly:
    ```bash
    python gui.py
    ```

2.  **Workflow**:
    - **Select Video**: Click "Select File" to choose a source video.
    - **Configure Settings**:
        - Choose Model (YOLO vs MediaPipe).
        - Adjust **Confidence**, **Edge Margin**, **Min Duration**, and **Max Angle**.
    - **Process**: Click **Start Processing**. The app will scan the video and populate the list of detected face segments.
    - **Review**: Click on any segment in the list to jump to that point in the player. Use "Preview Segment" to play just that clip.
    - **Export**: Select the clips you want (or "Select All") and click **Export Video** to render a new compiled video.

## Project Structure

- `gui.py`: Main entry point and GUI implementation.
- `processor.py`: Core logic for video processing, face detection, and video cutting.
- `requirements.txt`: List of Python dependencies.
- `run_facecut.bat`: Quick start script for Windows.
