import os
import signal
import subprocess
import json
from datetime import datetime
from pathlib import Path
from autonote.logger import log_info, log_error, log_warn, log_success
from autonote.config import config

def get_pactl_sources():
    try:
        mic_source = config.get("MIC_SOURCE", "").strip()
        system_source = config.get("SYSTEM_SOURCE", "").strip()
        
        if mic_source:
            default_source = mic_source
            log_info(f"Using custom MIC_SOURCE: {mic_source}")
        else:
            default_source = subprocess.check_output(["pactl", "get-default-source"]).decode().strip()
        
        if system_source:
            monitor_source = system_source
            log_info(f"Using custom SYSTEM_SOURCE: {system_source}")
        else:
            default_sink = subprocess.check_output(["pactl", "get-default-sink"]).decode().strip()
            monitor_source = f"{default_sink}.monitor"
            
            sources_list = subprocess.check_output(["pactl", "list", "sources", "short"]).decode()
            has_monitor = monitor_source in sources_list
            monitor_source = monitor_source if has_monitor else None
        
        return default_source, monitor_source
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to query pactl for audio sources. Ensure pulseaudio-utils is installed.")
        
def record_audio(duration: int = None, output_file: str = None, title: str = "") -> str:
    recordings_dir = Path(config["RECORDINGS_DIR"])
    now = datetime.now()
    date_str = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    if not output_file:
        output_file = recordings_dir / date_str / f"meeting_{timestamp}" / f"meeting_{timestamp}.wav"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        default_source, monitor_source = get_pactl_sources()
    except RuntimeError as e:
        log_error(str(e))
        return None
        
    log_info(f"Recording audio to {output_path}")
    
    cmd = ["ffmpeg", "-y"]
    if monitor_source:
        log_info("Mode: mic + system audio")
        mic_vol = config.get("MIC_VOLUME", "2.0")
        cmd.extend([
            "-f", "pulse", "-i", default_source,
            "-f", "pulse", "-i", monitor_source,
            "-filter_complex", f"[0:a]volume={mic_vol}[mic];[1:a][mic]amix=inputs=2:duration=longest:normalize=0"
        ])
    else:
        log_warn("Monitor source not found, recording mic only")
        cmd.extend(["-f", "pulse", "-i", default_source])
        
    if duration:
        cmd.extend(["-t", str(duration)])
        
    cmd.extend(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(output_path)])
    
    log_info("Press Ctrl+C to stop recording." if not duration else f"Recording for {duration} seconds.")
    
    # Use Popen for proper signal handling — lets us send 'q' to ffmpeg
    # for graceful shutdown so it finalizes the WAV headers.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        # Prevent Python's default SIGINT handler from killing the child
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )
    
    try:
        proc.wait()
    except KeyboardInterrupt:
        # Send 'q' for graceful ffmpeg shutdown (finalizes WAV headers)
        try:
            proc.stdin.write(b"q")
            proc.stdin.flush()
        except (BrokenPipeError, OSError):
            proc.terminate()
        # Give ffmpeg time to finalize the file
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            
    if not output_path.exists() or output_path.stat().st_size <= 78:
        # 78 bytes = empty WAV header with no audio data
        log_error("Recording file was not created or is empty. Please check the ffmpeg output above for details.")
        return None
        
    log_success(f"Recording saved to {output_path}")
    
    if not title:
        try:
            title = input("Meeting title (optional): ")
        except (KeyboardInterrupt, EOFError):
            title = ""
            print() # Print newline after ctrl+c input
            
    metadata = {
        "title": title,
        "timestamp": timestamp,
        "date": date_str,
        "audio_file": output_path.name,
        "created_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "local_recording"
    }
    
    meta_file = output_path.parent / f"{output_path.stem}_metadata.json"
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)
        
    return str(output_path)

# Alias to match naming convention of other audio modules
run_record = record_audio
