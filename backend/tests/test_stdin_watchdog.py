"""The stdin watchdog: the backend must exit when its parent app dies.

The menu-bar app spawns the backend holding the write end of its stdin
pipe (JARVIS_EXIT_ON_STDIN_CLOSE=1). If the app is force-killed the pipe
closes, and the backend must shut down instead of leaking on the port
with an auth token no future session knows.
"""

from __future__ import annotations

import os
import threading

from app.main import watch_stdin_and_terminate


def test_eof_triggers_shutdown_callback() -> None:
    read_fd, write_fd = os.pipe()
    fired = threading.Event()
    thread = watch_stdin_and_terminate(
        stream=os.fdopen(read_fd, "rb"), on_eof=fired.set
    )
    os.close(write_fd)  # the "parent" dies
    assert fired.wait(timeout=5), "watchdog did not react to stdin EOF"
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_data_before_eof_is_discarded_without_firing() -> None:
    read_fd, write_fd = os.pipe()
    fired = threading.Event()
    watch_stdin_and_terminate(stream=os.fdopen(read_fd, "rb"), on_eof=fired.set)
    os.write(write_fd, b"noise the app might emit\n")
    assert not fired.wait(timeout=0.2), "watchdog fired while the parent was alive"
    os.close(write_fd)
    assert fired.wait(timeout=5)
