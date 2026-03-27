import os
import subprocess
from pathlib import Path
from autonote.logger import log_info, log_error, log_success, log_warn

def compress_audio(wav_file: str, bitrate: str = "128k", delete_wav: bool = False) -> str:
    if not os.path.exists(wav_file):
        raise FileNotFoundError(f"Audio file not found: {wav_file}")
        
    wav_path = Path(wav_file)
    output_path = wav_path.with_suffix(".mp3")
    
    if output_path.exists():
        log_warn(f"Output file already exists: {output_path}")
        return str(output_path)
        
    log_info(f"Compressing: {wav_file} to {output_path} at {bitrate}")
    
    cmd = [
        "ffmpeg", "-i", str(wav_path),
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        "-y", str(output_path)
    ]
    
    try:
        # Run ffmpeg
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_success("Compression complete!")
        
        if delete_wav:
            os.remove(wav_path)
            log_info(f"Deleted original WAV: {wav_file}")
            
        return str(output_path)
    except subprocess.CalledProcessError as e:
        log_error(f"Compression failed: {e}")
        if output_path.exists():
            output_path.unlink()
        raise RuntimeError("FFMPEG compression failed")
