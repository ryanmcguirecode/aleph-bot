"""Keyboard control for data collection recording."""
import threading
from pynput import keyboard as pkb


class KeyboardControl:
    """Simple keyboard control for recording."""
    def __init__(self):
        self.exit_requested = False
        self.save_episode = False
        self.rerecord_episode = False
        self.delete_last_episode = False
        self.exit_early = False
        self._lock = threading.Lock()
        self._listener = None
    
    def _on_press(self, key):
        try:
            # Escape key - exit recording
            if key == pkb.Key.esc:
                with self._lock:
                    self.exit_requested = True
                    self.exit_early = True
                return False
            
            # Right arrow - save episode
            elif key == pkb.Key.right:
                with self._lock:
                    self.save_episode = True
                    self.exit_early = True
            
            # Left arrow - re-record current episode
            elif key == pkb.Key.left:
                with self._lock:
                    self.rerecord_episode = True
                    self.exit_early = True
            
            # Backspace - skip current episode (cancel recording)
            elif key == pkb.Key.backspace:
                with self._lock:
                    self.delete_last_episode = True
                    self.exit_early = True  # Also exit early to cancel current recording
        except AttributeError:
            pass
    
    def start(self):
        """Start keyboard listener."""
        self._listener = pkb.Listener(on_press=self._on_press)
        self._listener.start()
    
    def stop(self):
        """Stop keyboard listener."""
        if self._listener:
            self._listener.stop()
    
    def clear_flags(self):
        """Clear all flags except exit_requested."""
        with self._lock:
            self.save_episode = False
            self.rerecord_episode = False
            self.delete_last_episode = False
            self.exit_early = False
    
    def get_and_clear_flag(self, flag_name: str) -> bool:
        """Get and clear a specific flag."""
        with self._lock:
            value = getattr(self, flag_name, False)
            if value:
                setattr(self, flag_name, False)
            return value

