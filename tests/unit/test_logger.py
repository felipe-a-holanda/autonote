import pytest
from unittest.mock import patch, MagicMock
from autonote.logger import log_info, log_success, log_error, log_warn, log_debug, set_quiet


class TestLogger:
    
    @patch("autonote.logger.console")
    def test_log_info(self, mock_console):
        log_info("Test info message")
        mock_console.print.assert_called_once_with("[info][INFO][/info] Test info message")
    
    @patch("autonote.logger.console")
    def test_log_success(self, mock_console):
        log_success("Test success message")
        mock_console.print.assert_called_once_with("[success][SUCCESS][/success] Test success message")
    
    @patch("autonote.logger.console")
    def test_log_error(self, mock_console):
        log_error("Test error message")
        mock_console.print.assert_called_once_with("[error][ERROR][/error] Test error message")
    
    @patch("autonote.logger.console")
    def test_log_warn(self, mock_console):
        log_warn("Test warning message")
        mock_console.print.assert_called_once_with("[warn][WARN][/warn] Test warning message")
    
    @patch("autonote.logger.console")
    @patch("autonote.logger.config", {"DEBUG": "true"})
    def test_log_debug_enabled(self, mock_console):
        log_debug("Test debug message")
        mock_console.print.assert_called_once_with("[debug][DEBUG] Test debug message[/debug]")
    
    @patch("autonote.logger.console")
    @patch("autonote.logger.config", {"DEBUG": "false"})
    def test_log_debug_disabled(self, mock_console):
        log_debug("Test debug message")
        mock_console.print.assert_not_called()
    
    @patch("autonote.logger.console")
    def test_set_quiet_suppresses_info(self, mock_console):
        set_quiet(True)
        log_info("Should not print")
        mock_console.print.assert_not_called()
        set_quiet(False)
    
    @patch("autonote.logger.console")
    def test_set_quiet_suppresses_success(self, mock_console):
        set_quiet(True)
        log_success("Should not print")
        mock_console.print.assert_not_called()
        set_quiet(False)
    
    @patch("autonote.logger.console")
    def test_set_quiet_does_not_suppress_error(self, mock_console):
        set_quiet(True)
        log_error("Should still print")
        mock_console.print.assert_called_once_with("[error][ERROR][/error] Should still print")
        set_quiet(False)
    
    @patch("autonote.logger.console")
    def test_set_quiet_does_not_suppress_warn(self, mock_console):
        set_quiet(True)
        log_warn("Should still print")
        mock_console.print.assert_called_once_with("[warn][WARN][/warn] Should still print")
        set_quiet(False)
