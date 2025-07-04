# Different Python based programs for command line video editing

Install Python packages
pip install opencv-python numpy tqdm opencv-contrib-python

Install FFmpeg for audio support
Mac:
brew install ffmpeg

Windows:
Download from https://ffmpeg.org/download.html

Linux:
sudo apt update && sudo apt install ffmpeg

Install dependencies first for clean_video.py:
python clean_video.py --install-deps

Recreating the Environment:
List the packages in your current environment: pip freeze > requirements.txt.
Create a new virtual environment: python -m venv <new_env_name>.
Activate the new environment: source <new_env_name>/bin/activate.
Install the packages from the requirements file: pip install -r requirements.txt.
