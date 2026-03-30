import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestCompressAudio:
    
    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    @patch("pathlib.Path.exists", return_value=False)
    def test_compress_audio_success(self, mock_path_exists, mock_os_exists, mock_run):
        from autonote.audio.compress import compress_audio
        
        result = compress_audio("/tmp/test.wav", bitrate="128k")
        
        assert result == "/tmp/test.mp3"
        mock_run.assert_called_once()
        assert mock_run.call_args[0][0][0] == "ffmpeg"
    
    @patch("os.path.exists", return_value=False)
    def test_compress_audio_file_not_found(self, mock_exists):
        from autonote.audio.compress import compress_audio
        
        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            compress_audio("/tmp/nonexistent.wav")
    
    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    @patch("pathlib.Path.exists", return_value=True)
    def test_compress_audio_output_exists(self, mock_path_exists, mock_os_exists, mock_run):
        from autonote.audio.compress import compress_audio
        
        result = compress_audio("/tmp/test.wav")
        
        assert result == "/tmp/test.mp3"
        mock_run.assert_not_called()
    
    @patch("subprocess.run")
    @patch("os.remove")
    @patch("os.path.exists", return_value=True)
    @patch("pathlib.Path.exists", return_value=False)
    def test_compress_audio_delete_wav(self, mock_path_exists, mock_os_exists, mock_remove, mock_run):
        from autonote.audio.compress import compress_audio
        
        result = compress_audio("/tmp/test.wav", delete_wav=True)
        
        assert result == "/tmp/test.mp3"
        mock_remove.assert_called_once_with(Path("/tmp/test.wav"))
    
    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    @patch("pathlib.Path.exists", return_value=False)
    def test_compress_audio_ffmpeg_failure(self, mock_path_exists, mock_os_exists, mock_run):
        from autonote.audio.compress import compress_audio
        import subprocess
        
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")
        
        with pytest.raises(RuntimeError, match="FFMPEG compression failed"):
            compress_audio("/tmp/test.wav")
