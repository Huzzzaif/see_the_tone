# train/prepare_data.py
import argparse
from core.vad import detect_speech_intervals

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=str)
    ap.add_argument("--aggr", type=int, default=2)
    ap.add_argument("--frame_ms", type=int, default=30)
    ap.add_argument("--padding_ms", type=int, default=300)
    ap.add_argument("--min_speech_ms", type=int, default=200)
    ap.add_argument("--merge_gap_ms", type=int, default=200)
    args = ap.parse_args()

    ivals = detect_speech_intervals(
        args.audio,
        aggressiveness=args.aggr,
        frame_ms=args.frame_ms,
        padding_ms=args.padding_ms,
        min_speech_ms=args.min_speech_ms,
        merge_gap_ms=args.merge_gap_ms,
    )
    total = sum(e - s for s, e in ivals)
    print(f"Detected {len(ivals)} segments, total voiced {total:.2f}s")
    for i, (s, e) in enumerate(ivals, 1):
        print(f"{i:02d}. {s:.2f}s → {e:.2f}s  (dur {(e-s):.2f}s)")

if __name__ == "__main__":
    main()
