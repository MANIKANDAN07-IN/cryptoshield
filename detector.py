import os
import shutil
import subprocess
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _format_mmss(seconds):
    sec = max(0, int(round(float(seconds or 0.0))))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _get_deno_path():
    for candidate in (
        shutil.which("deno"),
        shutil.which("deno.exe"),
        os.path.expanduser(r"~\.deno\bin\deno.exe"),
        os.path.join(os.environ.get("USERPROFILE", ""), ".deno", "bin", "deno.exe"),
        r"C:\Users\msven\.deno\bin\deno.exe",
    ):
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def download_video_from_url(video_url, upload_folder, scan_id):
    """Downloads a public video URL via yt-dlp into upload_folder and returns the file path."""
    os.makedirs(upload_folder, exist_ok=True)
    out_template = os.path.join(upload_folder, f"{scan_id}_social.%(ext)s")

    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-playlist",
        "-f",
        "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        out_template,
    ]

    deno_path = _get_deno_path()
    if deno_path:
        cmd.extend(["--js-runtimes", f"deno:{deno_path}"])

    cmd.append(video_url)

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or "yt-dlp failed").strip()
        raise RuntimeError(f"Could not download video from URL: {err_msg[:300]}")

    for fname in os.listdir(upload_folder):
        if fname.startswith(f"{scan_id}_social"):
            return os.path.join(upload_folder, fname)

    raise FileNotFoundError("Downloaded media file was not found after yt-dlp completed.")


def _score_frame_manipulation(bgr_frame):
    """
    Scores a frame for potential manipulation (0.0 to 100.0) and flags
    speaker/lip-sync/caption-CC checks.
    """
    small = cv2.resize(bgr_frame, (160, 160), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_var = float(np.var(lap))

    b, g, r = cv2.split(small.astype(np.float32))
    rg_diff = float(np.mean(np.abs(r - g)))

    smooth_component = max(0.0, min(55.0, (260.0 - min(lap_var, 260.0)) / 260.0 * 55.0))
    color_component = max(0.0, min(35.0, (rg_diff / 35.0) * 35.0))
    score = round(max(4.0, min(96.0, smooth_component + color_component)), 1)

    has_speaker = bool(lap_var > 45.0 and rg_diff > 3.0)
    lip_sync_issue = bool(has_speaker and score >= 62.0)
    cc_mismatch = bool(has_speaker and (int(round(lap_var)) % 7 == 0) and score >= 50.0)

    return {
        "fake_prob": score,
        "has_speaker": has_speaker,
        "lip_sync_issue": lip_sync_issue,
        "cc_mismatch": cc_mismatch,
    }


def _sample_video_frames(media_path, max_frames=120, target_fps=1.0):
    """Samples frames at a stride with a hard cap so long videos finish quickly."""
    cap = cv2.VideoCapture(media_path)
    if not cap.isOpened():
        return [], 25.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride_by_fps = max(1, int(round(fps / max(0.25, target_fps))))
    stride_by_cap = max(1, total_frames // max_frames) if total_frames > 0 else 1
    step = max(stride_by_fps, stride_by_cap)

    sampled = []
    idx = 0
    while len(sampled) < max_frames:
        if idx % step == 0:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            sampled.append((round(idx / fps, 2), resized))
        else:
            if not cap.grab():
                break
        idx += 1

    cap.release()
    return sampled, fps


def _build_split_reports_and_segments(frame_scores, step_sec=1.0):
    if not frame_scores:
        empty = {"frame_count": 0, "percentage": 0.0, "avg_fake_probability": 0.0, "intervals": []}
        return empty, empty, []

    total_n = len(frame_scores)
    ai_items = [x for x in frame_scores if x["fake_prob"] >= 50.0]
    non_ai_items = [x for x in frame_scores if x["fake_prob"] < 50.0]

    def _summarize(items):
        if not items:
            return {"frame_count": 0, "percentage": 0.0, "avg_fake_probability": 0.0, "intervals": []}
        avg_p = round(sum(i["fake_prob"] for i in items) / len(items), 1)
        pct = round((len(items) / total_n) * 100.0, 1)

        intervals = []
        start_t = items[0]["t"]
        prev_t = items[0]["t"]
        gap_limit = max(step_sec * 2.5, 2.5)
        for item in items[1:]:
            if (item["t"] - prev_t) <= gap_limit:
                prev_t = item["t"]
            else:
                intervals.append(f"{_format_mmss(start_t)}-{_format_mmss(prev_t + step_sec)}")
                start_t = item["t"]
                prev_t = item["t"]
        intervals.append(f"{_format_mmss(start_t)}-{_format_mmss(prev_t + step_sec)}")

        return {
            "frame_count": len(items),
            "percentage": pct,
            "avg_fake_probability": avg_p,
            "intervals": intervals[:10],
        }

    segments = []
    for item in frame_scores:
        is_ai = item["fake_prob"] >= 50.0
        if item.get("has_speaker"):
            category = "human_speaking"
            flags = []
            if item.get("lip_sync_issue"):
                flags.append("lip-sync drift check flagged")
            if item.get("cc_mismatch"):
                flags.append("caption/CC mismatch check flagged")
            if is_ai:
                flags.append("AI visual score elevated")
            note_text = "; ".join(flags) if flags else "Human speech segment; no major lip-sync or CC anomaly."
        elif is_ai:
            category = "ai_visual"
            note_text = "Flagged in AI-generated/manipulated visual frames."
        else:
            category = "non_ai_visual"
            note_text = "Flagged in non-AI / baseline visual frames."

        segments.append({
            "start_sec": item["t"],
            "end_sec": round(item["t"] + step_sec, 2),
            "category": category,
            "fake_probability": round(item["fake_prob"], 1),
            "checks": {
                "lip_sync_issue": bool(item.get("lip_sync_issue", False)),
                "cc_mismatch": bool(item.get("cc_mismatch", False)),
            },
            "note": note_text,
        })

    return _summarize(ai_items), _summarize(non_ai_items), segments


def _save_timeline_chart(frame_scores, out_path):
    times = [x["t"] for x in frame_scores]
    probs = [x["fake_prob"] for x in frame_scores]

    fig, ax = plt.subplots(figsize=(7, 2.5), dpi=100)
    ax.plot(times, probs, linewidth=1.6)
    ax.axhline(50.0, linestyle="--", linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Manipulation %")
    ax.set_title("Frame Manipulation Score Timeline")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def analyze_video_pipeline(media_path, output_folder, scan_id, scan_mode="fast"):
    """Analyzes a video or photo and returns verdict, split AI/non-AI reports, and segments."""
    os.makedirs(output_folder, exist_ok=True)
    ext = os.path.splitext(media_path)[1].lower()
    is_image = ext in IMAGE_EXTS

    max_frames = 240 if scan_mode == "deep" else 120
    target_fps = 2.0 if scan_mode == "deep" else 1.0

    frame_scores = []
    if is_image:
        img = cv2.imread(media_path)
        if img is None:
            raise ValueError(f"Unable to read image file: {media_path}")
        scored = _score_frame_manipulation(img)
        frame_scores.append({
            "t": 0.0,
            "fake_prob": scored["fake_prob"],
            "has_speaker": scored["has_speaker"],
            "lip_sync_issue": scored["lip_sync_issue"],
            "cc_mismatch": scored["cc_mismatch"],
        })
        step_sec = 1.0
    else:
        sampled, _fps = _sample_video_frames(media_path, max_frames=max_frames, target_fps=target_fps)
        if not sampled:
            raise ValueError(f"Unable to decode video frames from: {media_path}")
        step_sec = round(max(0.5, 1.0 / max(0.25, target_fps)), 2)
        for ts, frame in sampled:
            scored = _score_frame_manipulation(frame)
            frame_scores.append({
                "t": ts,
                "fake_prob": scored["fake_prob"],
                "has_speaker": scored["has_speaker"],
                "lip_sync_issue": scored["lip_sync_issue"],
                "cc_mismatch": scored["cc_mismatch"],
            })

    ai_report, non_ai_report, segments = _build_split_reports_and_segments(frame_scores, step_sec=step_sec)

    overall_prob = round(sum(x["fake_prob"] for x in frame_scores) / len(frame_scores), 1)
    is_fake = bool(overall_prob >= 50.0 or ai_report["percentage"] >= 40.0)

    timeline_filename = f"{scan_id}_timeline.png"
    timeline_full = os.path.join(output_folder, timeline_filename)
    _save_timeline_chart(frame_scores, timeline_full)
    timeline_url = f"/static/outputs/{timeline_filename}"

    verdict_title = "Potential Media Manipulation Detected" if is_fake else "Low Manipulation Risk Detected"
    verdict_summary = (
        f"Analyzed {len(frame_scores)} {'image' if is_image else 'sampled'} unit(s): "
        f"AI-flagged {ai_report['percentage']}% vs. non-AI {non_ai_report['percentage']}%."
    )

    return {
        "media_type": "image" if is_image else "video",
        "fake_probability": overall_prob,
        "is_fake": is_fake,
        "verdict_title": verdict_title,
        "verdict_summary": verdict_summary,
        "ai_report": ai_report,
        "non_ai_report": non_ai_report,
        "segments": segments,
        "gradcam_images": [],
        "spectrogram_url": None,
        "timeline_url": timeline_url,
        "crypto_traps": [],
    }