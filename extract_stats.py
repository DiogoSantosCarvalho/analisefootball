import sys
import json
from pathlib import Path


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def summarize(export: dict):
    out = {}
    out['video'] = export.get('video')
    out['total_time_s'] = export.get('total_time_s')
    out['frames_processed'] = export.get('frames_processed')
    out['avg_time_per_frame_s'] = export.get('avg_time_per_frame_s')

    # requested metrics block from main.py
    requested = export.get('requested_metrics', {}) or {}
    fps_block = requested.get('fps', {}) or {}
    det_block = requested.get('detection_metrics', {}) or {}
    trk_block = requested.get('tracking_metrics', {}) or {}
    out['processing_fps'] = fps_block.get('processing_fps')
    out['source_video_fps'] = fps_block.get('source_video_fps')
    out['real_time_capable'] = fps_block.get('real_time_capable')
    out['precision'] = det_block.get('precision')
    out['recall'] = det_block.get('recall')
    out['mAP'] = det_block.get('mAP')
    out['MOTA'] = trk_block.get('MOTA')
    out['IDF1'] = trk_block.get('IDF1')
    out['detection_metrics_status'] = det_block.get('status')
    out['tracking_metrics_status'] = trk_block.get('status')

    # locked teams
    locked = export.get('locked_teams', {}) or {}
    out['num_locked_teams'] = len(locked)
    team_counts = {}
    for tid, team in locked.items():
        team_counts[str(team)] = team_counts.get(str(team), 0) + 1
    out['team_counts'] = team_counts

    # distances and speeds
    distances = export.get('player_distances_m', {}) or {}
    speeds = export.get('player_speeds_kmh', {}) or {}
    # convert keys to str and values to float
    dist_items = [(str(k), float(v)) for k, v in distances.items()]
    speed_items = [(str(k), float(v)) for k, v in speeds.items()]
    dist_sorted = sorted(dist_items, key=lambda x: x[1], reverse=True)
    speed_sorted = sorted(speed_items, key=lambda x: x[1], reverse=True)

    out['top5_distances'] = dist_sorted[:5]
    out['top5_speeds'] = speed_sorted[:5]

    # tracks summary: count frames per tracker and fragmentation
    tracks = export.get('tracks', {}) or {}
    player_tracks = tracks.get('player', {}) or {}
    # player_tracks: {frame_idx: {tracker_id: [x1,y1,x2,y2]}}
    tracker_frames = {}
    for frame_str, d in player_tracks.items():
        # frame_str might be string or int
        try:
            frame_idx = int(frame_str)
        except Exception:
            frame_idx = frame_str
        for tid in d.keys():
            tracker_frames.setdefault(str(tid), []).append(int(frame_idx))

    tracker_stats = {}
    for tid, frames in tracker_frames.items():
        frames_sorted = sorted(frames)
        # count continuous fragments
        fragments = 0
        last = None
        for f in frames_sorted:
            if last is None or f - last > 1:
                fragments += 1
            last = f
        tracker_stats[tid] = {'frames_present': len(frames_sorted), 'fragments': fragments}

    # overall stats
    out['num_unique_trackers'] = len(tracker_stats)
    # average frames per tracker
    if tracker_stats:
        out['avg_frames_per_tracker'] = sum(v['frames_present'] for v in tracker_stats.values()) / len(tracker_stats)
    else:
        out['avg_frames_per_tracker'] = 0

    # fragmentation distribution
    frag_counts = {}
    for v in tracker_stats.values():
        frag_counts[str(v['fragments'])] = frag_counts.get(str(v['fragments']), 0) + 1
    out['fragmentation_distribution'] = frag_counts

    return out


def main():
    # find json file
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    else:
        # auto-find *_data.json in current dir
        files = list(Path('.').glob('*_data.json'))
        if not files:
            print('No _data.json found in current directory. Pass path as argument.')
            sys.exit(1)
        path = files[0]

    export = load_json(path)
    summary = summarize(export)

    print('\n=== Analysis Summary ===')
    print(f"Video: {summary.get('video')}")
    print(f"Total time (s): {summary.get('total_time_s')}")
    print(f"Frames processed: {summary.get('frames_processed')}")
    print(f"Avg time/frame (s): {summary.get('avg_time_per_frame_s')}")
    print(f"Source FPS: {summary.get('source_video_fps')}")
    print(f"Processing FPS: {summary.get('processing_fps')}")
    print(f"Real-time capable: {summary.get('real_time_capable')}")
    print(f"Precision: {summary.get('precision')}")
    print(f"Recall: {summary.get('recall')}")
    print(f"mAP: {summary.get('mAP')}")
    print(f"MOTA: {summary.get('MOTA')}")
    print(f"IDF1: {summary.get('IDF1')}")
    print(f"Detection metrics status: {summary.get('detection_metrics_status')}")
    print(f"Tracking metrics status: {summary.get('tracking_metrics_status')}")
    print(f"Unique trackers: {summary.get('num_unique_trackers')}")
    print(f"Avg frames per tracker: {summary.get('avg_frames_per_tracker'):.1f}")
    print(f"Top 5 distances (m): {summary.get('top5_distances')}")
    print(f"Top 5 speeds (km/h): {summary.get('top5_speeds')}")
    print(f"Team counts (locked): {summary.get('team_counts')}")
    print(f"Fragmentation distribution: {summary.get('fragmentation_distribution')}")


if __name__ == '__main__':
    main()
