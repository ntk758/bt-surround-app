"""
device_monitor.py
-----------------
出力ストリームの生存監視と、再生を止めずに行うデバイス接続確認。

PortAudio のデバイス一覧は再初期化しないと更新されず、再初期化すると再生中の
ストリームが壊れる。そのため接続確認には winmm を直接使う。
"""

from __future__ import annotations

import re
import sys
import threading
import time


class StreamWatch:
    """ストリームごとに最後のコールバック時刻を記録し、止まったものを検出する。"""

    def __init__(self, stale_after_s: float = 3.0, startup_grace_s: float = 5.0):
        self.stale_after_s = stale_after_s
        self.startup_grace_s = startup_grace_s
        self._lock = threading.Lock()
        self._streams: dict[str, object] = {}
        self._last: dict[str, float | None] = {}
        self._started: dict[str, float] = {}

    def register(self, role: str, stream) -> None:
        with self._lock:
            self._streams[role] = stream
            self._last[role] = None
            self._started[role] = time.monotonic()

    def touch(self, role: str) -> None:
        """オーディオコールバックから呼ぶ。"""
        self._last[role] = time.monotonic()

    def lost(self) -> list[str]:
        """停止したストリームの役割名を返す。"""
        now = time.monotonic()
        result = []
        with self._lock:
            for role, stream in self._streams.items():
                if not getattr(stream, "active", True):
                    result.append(role)
                    continue
                last = self._last[role]
                if last is None:
                    # BTスピーカーは起動に時間がかかるので猶予を長めに取る
                    if now - self._started[role] > self.startup_grace_s:
                        result.append(role)
                elif now - last > self.stale_after_s:
                    result.append(role)
        return result


def _list_winmm_names() -> list[str] | None:
    """接続中の入出力デバイス名(最大31文字)を返す。Windows以外や失敗時は None。"""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class WAVEOUTCAPSW(ctypes.Structure):
            _fields_ = [("wMid", wintypes.WORD), ("wPid", wintypes.WORD), ("vDriverVersion", wintypes.UINT),
                        ("szPname", wintypes.WCHAR * 32), ("dwFormats", wintypes.DWORD),
                        ("wChannels", wintypes.WORD), ("wReserved1", wintypes.WORD), ("dwSupport", wintypes.DWORD)]

        class WAVEINCAPSW(ctypes.Structure):
            _fields_ = [("wMid", wintypes.WORD), ("wPid", wintypes.WORD), ("vDriverVersion", wintypes.UINT),
                        ("szPname", wintypes.WCHAR * 32), ("dwFormats", wintypes.DWORD),
                        ("wChannels", wintypes.WORD), ("wReserved1", wintypes.WORD)]

        winmm = ctypes.windll.winmm
        names = []
        for count_fn, caps_fn, caps_type in (
            (winmm.waveOutGetNumDevs, winmm.waveOutGetDevCapsW, WAVEOUTCAPSW),
            (winmm.waveInGetNumDevs, winmm.waveInGetDevCapsW, WAVEINCAPSW),
        ):
            for i in range(count_fn()):
                caps = caps_type()
                if caps_fn(i, ctypes.byref(caps), ctypes.sizeof(caps)) == 0:
                    names.append(caps.szPname)
        return names
    except Exception as e:
        print(f"[device_monitor] error: {e}", file=sys.stderr)
        return None


def device_name_from_display(display: str) -> str:
    """"[API] 名前 (ID: n)" から名前だけを取り出す。"""
    name = re.sub(r"\s*\(ID:\s*\d+\)$", "", display)
    return re.sub(r"^\[[^\]]*\]\s*", "", name).strip()


def device_is_present(display: str, live_names: list[str] | None = None) -> bool:
    """デバイスが接続されているか。確認手段がなければ True (再接続を試みる) を返す。

    MME の名前は31文字で切れ、Host API によって括弧の有無が変わることがあるので
    前方一致で比較する。
    """
    names = live_names if live_names is not None else _list_winmm_names()
    if names is None:
        return True
    target = device_name_from_display(display).lower()
    if not target:
        return False
    for live in names:
        live = live.lower().strip()
        if live and (target.startswith(live) or live.startswith(target)):
            return True
    return False
