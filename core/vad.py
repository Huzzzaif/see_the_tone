# core/vad.py
from __future__ import annotations
from typing import List, Tuple, Optional, Iterable
import collections
import math

import numpy as np
import librosa
import webrtcvad

def _float_to_pcm16(x: np.ndarray) -> bytes:
    """Convert float32 audio in [-1, 1] to little-endian PCM16 bytes."""
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    x = np.clip(x, -1.0, 1.0)
    x_i16 = (x * 32767.0).astype(np.int16)
    return x_i16.tobytes()

def _frame_generator(signal_pcm16: bytes, sample_rate: int, frame_ms: int) -> Iterable[Tuple[float, bytes]]:
    """
    Yield (start_time_sec, frame_bytes) for exact frame_ms windows, dropping any tail shorter than a full frame.
    """
    if frame_ms not in (10, 20, 30):
        raise ValueError("frame_ms must be one of {10, 20, 30} for WebRTC VAD")

    bytes_per_sample = 2  # int16
    frame_bytes = int(sample_rate * frame_ms / 1000) * bytes_per_sample
    if frame_bytes == 0:
        return

    offset = 0
    t = 0.0
    step_s = frame_ms / 1000.0
    total = len(signal_pcm16)
    while offset + frame_bytes <= total:
        yield t, signal_pcm16[offset: offset + frame_bytes]
        t += step_s
        offset += frame_bytes

def detect_speech_intervals(
    wav_path: str,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    padding_ms: int = 300,
    min_speech_ms: int = 200,
    merge_gap_ms: int = 200,
    target_sr: int = 16000,
) -> List[Tuple[float, float]]:
    """
    Run WebRTC VAD on an audio file and return cleaned speech intervals.

    Args:
        wav_path: Path to audio (wav/mp3/m4a). Will be resampled to mono 16 kHz.
        aggressiveness: 0..3 (higher = stricter, fewer false positives).
        frame_ms: Frame size for VAD (10|20|30 ms). 30 ms is a good default.
        padding_ms: Hysteresis window for start/stop smoothing.
        min_speech_ms: Drop segments shorter than this duration.
        merge_gap_ms: Merge consecutive segments separated by small gaps.
        target_sr: Sample rate used by VAD (must be 16000).

    Returns:
        List of (start_sec, end_sec) tuples for speech-only regions, sorted and non-overlapping.
    """
    #Check if the input is correct
    if aggressiveness not in(0,1,2,3):
        raise ValueError("aggressive value must be between 0 and 3")
    if frame_ms not in(10,20,30):
        raise ValueError("frame_ms must be 10, 20 or 30")
    if padding_ms < frame_ms:
        padding_ms = frame_ms
    # Implementation will:
    # 1) load + resample -> mono 16 kHz
    # librosa.load returns float32 in [-1, 1]
    y, sr=librosa.load(wav_path,sr=target_sr,mono=True)
    if y.size == 0:
        return []
    # 2) convert to PCM16 bytes
    pcm16 = _float_to_pcm16(y)
    # 3) slice into exact frames of `frame_ms`
    frames = list(_frame_generator(pcm16, target_sr, frame_ms))
    if not frames:
        return []
    # 4) call webrtcvad.is_speech per frame
    vad = webrtcvad.Vad(aggressiveness)
    voiced_flags = [vad.is_speech(frame_bytes , target_sr) for _, frame_bytes in frames]
    # 5) smooth with a ring buffer (padding_ms) to avoid flicker
    pad_frames = max(1, padding_ms // frame_ms)
    ring = collections.deque(maxlen=pad_frames)

    intervals: List[Tuple[float, float]] = []
    in_speech = False
    seg_start: Optional[float] = None

    def frac_true(buf: collections.deque) -> float:
        if not buf:
            return 0.0
        return sum(1 for b in buf if b) / len(buf)
    # 6) collapse to intervals, drop short, merge tiny gaps
    # thresholds with hysteresis (enter > 0.6, exit < 0.4)
    th_enter, th_exit = 0.6, 0.4

    for (t, _), is_voiced in zip(frames, voiced_flags):
        ring.append(is_voiced)

        # Enter speech
        if not in_speech and frac_true(ring) > th_enter:
            in_speech = True
            # Align start to earliest time covered by the ring buffer
            start_shift_s = len(ring) * (frame_ms / 1000.0)
            seg_start = max(0.0, t - start_shift_s)

        # Exit speech
        elif in_speech and frac_true(ring) < th_exit:
            in_speech = False
            seg_end = t
            if seg_start is not None and seg_end > seg_start:
                intervals.append((seg_start, seg_end))
            seg_start = None

    # Flush tail if ended inside speech
    if in_speech and seg_start is not None:
        total_dur = len(y) / float(target_sr)
        if total_dur > seg_start:
            intervals.append((seg_start, total_dur))

    # 6) Post-process: drop short segments, merge small gaps
    def ms(dur_s: float) -> float:
        return dur_s * 1000.0

    cleaned: List[Tuple[float, float]] = []
    for s, e in intervals:
        if ms(e - s) >= min_speech_ms:
            if cleaned and ms(s - cleaned[-1][1]) <= merge_gap_ms:
                # merge with previous
                ps, pe = cleaned[-1]
                cleaned[-1] = (ps, max(pe, e))
            else:
                cleaned.append((s, e))

    # Final sanity: monotonic, positive durations
    out: List[Tuple[float, float]] = []
    for s, e in cleaned:
        s = max(0.0, float(s))
        e = max(0.0, float(e))
        if e > s:
            out.append((s, e))

    return out