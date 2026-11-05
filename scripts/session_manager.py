import threading
import time


class SessionManager:
    """Manages an exclusive active try-on session for multi-device LAN access.

    Ensures that only one device streams to the single physical GPU at any given time,
    while allowing other devices to browse garments, upload images, or view status.
    """

    def __init__(self, timeout_seconds: float = 60.0):
        self._lock = threading.Lock()
        self.active_session_id: str | None = None
        self.active_client_ip: str | None = None
        self.session_start_time: float | None = None
        self.last_activity_time: float | None = None
        self.timeout_seconds = timeout_seconds

    def _check_timeout_locked(self) -> None:
        if self.active_session_id is not None and self.last_activity_time is not None:
            idle_time = time.time() - self.last_activity_time
            if idle_time > self.timeout_seconds:
                print(
                    f"[SessionManager] Active session {self.active_session_id} from {self.active_client_ip} "
                    f"timed out after {idle_time:.1f}s of inactivity. Lock released.",
                    flush=True,
                )
                self.active_session_id = None
                self.active_client_ip = None
                self.session_start_time = None
                self.last_activity_time = None

    def acquire(self, session_id: str, client_ip: str) -> bool:
        """Attempt to acquire exclusive GPU try-on session.

        Returns True if acquired (or re-acquired by the same session), False if another user holds the session.
        """
        with self._lock:
            self._check_timeout_locked()
            if self.active_session_id is None or self.active_session_id == session_id:
                self.active_session_id = session_id
                self.active_client_ip = client_ip
                now = time.time()
                if self.session_start_time is None:
                    self.session_start_time = now
                self.last_activity_time = now
                print(
                    f"[SessionManager] GPU session acquired by {client_ip} (session: {session_id})",
                    flush=True,
                )
                return True
            return False

    def touch(self, session_id: str) -> None:
        """Update last activity timestamp for the active session (called on frame arrival)."""
        with self._lock:
            if self.active_session_id == session_id:
                self.last_activity_time = time.time()

    def release(self, session_id: str) -> bool:
        """Release the exclusive GPU session."""
        with self._lock:
            if self.active_session_id == session_id:
                elapsed = int(time.time() - (self.session_start_time or time.time()))
                print(
                    f"[SessionManager] GPU session released by {self.active_client_ip} (duration: {elapsed}s)",
                    flush=True,
                )
                self.active_session_id = None
                self.active_client_ip = None
                self.session_start_time = None
                self.last_activity_time = None
                return True
            return False

    def force_release(self) -> None:
        """Force-release any active session (admin override)."""
        with self._lock:
            self.active_session_id = None
            self.active_client_ip = None
            self.session_start_time = None
            self.last_activity_time = None

    def get_status(self) -> dict:
        """Get the current session occupancy status."""
        with self._lock:
            self._check_timeout_locked()
            if self.active_session_id is not None and self.session_start_time is not None:
                elapsed = int(time.time() - self.session_start_time)
                return {
                    "in_use": True,
                    "active_ip": self.active_client_ip,
                    "elapsed_seconds": elapsed,
                    "timeout_seconds": int(self.timeout_seconds),
                }
            return {
                "in_use": False,
                "active_ip": None,
                "elapsed_seconds": 0,
                "timeout_seconds": int(self.timeout_seconds),
            }

