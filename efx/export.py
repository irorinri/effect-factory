"""ffmpeg discovery, encoder probing and clip export."""

import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import zipfile

from .core import TEMPLATES_DIR, ensure_dir, now_ts, write_json
from .engine import FrameRenderer, luma_alpha, render_sequence

FORMATS = {
    "mp4": {"label": "MP4 (H.264)", "ext": ".mp4", "alpha": False,
            "hint": "Black background. Use Screen / Add blend in your editor."},
    "mov_alpha": {"label": "MOV with alpha", "ext": ".mov", "alpha": True,
                  "hint": "ProRes 4444 with transparency derived from brightness."},
    "png": {"label": "PNG sequence", "ext": "", "alpha": True,
            "hint": "Numbered transparent PNG frames in a folder."},
}

QUALITIES = {
    # name: (x264 CRF, hardware CQ, bitrate at 1080p30 in Mbit/s)
    "draft": (24, 28, 8),
    "standard": (19, 23, 14),
    "high": (16, 19, 24),
    "max": (12, 15, 40),
}

ENCODER_LABELS = {
    "auto": "Auto",
    "libx264": "x264 (CPU)",
    "h264_nvenc": "NVIDIA NVENC",
    "h264_qsv": "Intel Quick Sync",
    "h264_amf": "AMD AMF",
    "h264_videotoolbox": "Apple VideoToolbox",
}
HARDWARE_ENCODERS = ("h264_nvenc", "h264_qsv", "h264_amf", "h264_videotoolbox")


def _no_window_flags():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_ffmpeg(preferred=None):
    """Locate an ffmpeg executable, or return None."""
    candidates = []
    if preferred:
        preferred = os.path.expanduser(str(preferred).strip().strip('"'))
        candidates.append(preferred)
    candidates.append("ffmpeg")
    exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates += [os.path.join(here, exe), os.path.join(here, "ffmpeg", "bin", exe)]
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        candidates += [r"C:\ffmpeg\bin\ffmpeg.exe", os.path.join(local, "Microsoft", "WinGet", "Links", "ffmpeg.exe"),
                       os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "ffmpeg", "bin", "ffmpeg.exe")]
    elif sys.platform == "darwin":
        candidates += ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
    else:
        candidates += ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/snap/bin/ffmpeg"]
    for cand in candidates:
        if not cand:
            continue
        if os.path.isfile(cand):
            return os.path.abspath(cand)
        found = shutil.which(cand)
        if found:
            return found
    try:  # optional dependency that bundles a static ffmpeg
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _run(cmd, timeout=10):
    return subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          timeout=timeout, creationflags=_no_window_flags())


def ffmpeg_version(ffmpeg):
    try:
        out = _run([ffmpeg, "-hide_banner", "-version"], timeout=6).stdout.decode("utf-8", "ignore")
        first = out.splitlines()[0] if out else ""
        return first.replace("ffmpeg version", "").strip().split(" ")[0] or "unknown"
    except Exception:
        return None


def probe_encoders(ffmpeg):
    """Return the H.264 encoders that actually work on this machine (best first)."""
    try:
        listing = _run([ffmpeg, "-hide_banner", "-encoders"], timeout=8).stdout.decode("utf-8", "ignore")
    except Exception:
        return []
    working = []
    for enc in (*HARDWARE_ENCODERS, "libx264"):
        if f" {enc} " not in listing:
            continue
        if enc == "libx264":
            working.append(enc)
            continue
        if enc == "h264_videotoolbox" and sys.platform != "darwin":
            continue
        try:
            res = _run([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                        "color=c=black:s=256x144:d=0.2", "-frames:v", "2", "-c:v", enc, "-f", "null", "-"], timeout=12)
            if res.returncode == 0:
                working.append(enc)
        except Exception:
            continue
    if "libx264" not in working and " libopenh264 " in listing:
        working.append("libopenh264")
    return working


def pick_encoder(requested, available):
    if requested and requested != "auto" and requested in available:
        return requested
    for enc in (*HARDWARE_ENCODERS, "libx264", "libopenh264"):
        if enc in available:
            return enc
    return "libx264"


def _bitrate(quality, w, h, fps):
    base = QUALITIES.get(quality, QUALITIES["standard"])[2]
    scale = (w * h * max(1, fps)) / float(1920 * 1080 * 30)
    return f"{max(2.0, base * max(0.35, scale)):.1f}M"


def video_codec_args(encoder, quality, w, h, fps):
    crf, cq, _ = QUALITIES.get(quality, QUALITIES["standard"])
    rate = _bitrate(quality, w, h, fps)
    if encoder == "libx264":
        preset = "slow" if quality in ("high", "max") else ("veryfast" if quality == "draft" else "medium")
        return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-tune", "film"]
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", rate,
                "-maxrate", rate, "-bufsize", rate]
    if encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", str(cq), "-b:v", rate]
    if encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-usage", "transcoding", "-quality", "quality", "-rc", "vbr_peak", "-b:v", rate,
                "-maxrate", rate]
    if encoder == "h264_videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-b:v", rate, "-allow_sw", "1"]
    return ["-c:v", encoder, "-b:v", rate]


def build_command(ffmpeg, fmt, encoder, quality, w, h, fps, out_path):
    if fmt == "mov_alpha":
        return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an", "-c:v", "prores_ks", "-profile:v", "4444",
                "-pix_fmt", "yuva444p10le", "-vendor", "apl0", out_path]
    # BT.709 tagging keeps colours consistent in editors; pure black stays black.
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-an",
            *video_codec_args(encoder, quality, w, h, fps),
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-movflags", "+faststart", out_path]


def format_command(cmd):
    return " ".join(f'"{c}"' if (" " in str(c) or not c) else str(c) for c in cmd)


def safe_name(text):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(text).strip()]
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "overlay"


def output_base(prefix, look_name, effect_id, w, h, fps, suffix=""):
    return f"{safe_name(prefix)}_{safe_name(look_name)}_{w}x{h}_{fps}fps_{now_ts()}{suffix}"


class ExportCancelled(Exception):
    pass


def export_clip(spec, *, ffmpeg, out_dir, base_name, fmt="mp4", encoder="libx264", quality="standard",
                metadata=None, progress=None, cancel=None, workers=None, log=None):
    """Render a clip and write it (plus a thumbnail and JSON metadata).

    Returns a dict of output paths.  Raises ExportCancelled when cancelled.
    """
    progress = progress or (lambda frac, info=None: None)
    log = log or (lambda msg: None)
    cancel = cancel or threading.Event()
    fmt = fmt if fmt in FORMATS else "mp4"
    ensure_dir(out_dir)
    renderer = FrameRenderer(spec)
    frames = spec.frames
    w, h, fps = int(spec.w), int(spec.h), int(spec.fps)
    if workers is None:
        workers = max(1, min(4, (os.cpu_count() or 2) - 1))
    info = FORMATS[fmt]
    thumb_path = os.path.join(out_dir, base_name + "_thumb.png")
    meta_path = os.path.join(out_dir, base_name + ".json")
    started = time.perf_counter()
    proc = None
    out_path = None
    try:
        if fmt == "png":
            out_path = ensure_dir(os.path.join(out_dir, base_name))
        else:
            out_path = os.path.join(out_dir, base_name + info["ext"])
            cmd = build_command(ffmpeg, fmt, encoder, quality, w, h, fps, out_path)
            log("ffmpeg: " + format_command(cmd))
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    creationflags=_no_window_flags())
        first = None
        for n, (i, img) in enumerate(render_sequence(renderer, range(frames), workers=workers, cancel=cancel)):
            if cancel.is_set():
                raise ExportCancelled()
            if first is None:
                first = img
            if fmt == "png":
                luma_alpha(img).save(os.path.join(out_path, f"frame_{i + 1:05d}.png"), optimize=False)
            elif fmt == "mov_alpha":
                proc.stdin.write(luma_alpha(img).tobytes())
            else:
                proc.stdin.write(img.tobytes())
            elapsed = time.perf_counter() - started
            done = n + 1
            eta = elapsed / done * (frames - done)
            progress(done / float(frames), {"frame": done, "frames": frames, "eta": eta, "elapsed": elapsed})
        if cancel.is_set():
            raise ExportCancelled()
        if proc is not None:
            proc.stdin.close()
            err = proc.stderr.read().decode("utf-8", "ignore") if proc.stderr else ""
            code = proc.wait()
            proc.stderr.close()
            proc = None
            if code != 0:
                raise RuntimeError(f"ffmpeg failed (exit code {code}).\n{err[-1500:]}")
        if first is not None:
            first.save(thumb_path)
        meta = dict(metadata or {})
        meta.update({
            "outputs": {"video": out_path, "thumb": thumb_path, "format": fmt},
            "render": {"w": w, "h": h, "fps": fps, "duration": float(spec.duration), "frames": frames,
                       "loop": bool(spec.loop), "native_loop": renderer.native_loop,
                       "crossfade_frames": renderer.crossfade_frames, "encoder": encoder if fmt == "mp4" else fmt,
                       "quality": quality, "seconds": round(time.perf_counter() - started, 2)},
            "created": now_ts(),
            "note": "Black background overlay. Use Screen / Add blend." if fmt == "mp4" else "Transparent overlay.",
        })
        write_json(meta_path, meta)
        return {"video": out_path, "thumb": thumb_path, "meta": meta_path}
    except BaseException:
        if proc is not None:
            for stream in (proc.stdin, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass
            try:
                proc.kill()
                proc.wait(timeout=5)
            except Exception:
                pass
        if out_path and fmt != "png" and os.path.isfile(out_path) and cancel.is_set():
            try:
                os.remove(out_path)
            except OSError:
                pass
        raise


def create_package_zip(video_path, templates_dir=TEMPLATES_DIR):
    """Bundle a render with README/LICENSE templates (for asset stores)."""
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError("Export a video first.")

    def template(name, fallback):
        try:
            with open(os.path.join(templates_dir, name), "r", encoding="utf-8") as f:
                text = f.read()
            return text if text.strip() else fallback
        except Exception:
            return fallback

    base, ext = os.path.splitext(video_path)
    zip_path = base + "__package.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if os.path.isdir(video_path):
            for fn in sorted(os.listdir(video_path)):
                zf.write(os.path.join(video_path, fn), arcname=f"frames/{fn}")
        else:
            zf.write(video_path, arcname="overlay" + ext)
        zf.writestr("README.txt", template("README.txt", "Overlay Video Asset\n"))
        zf.writestr("LICENSE.txt", template("LICENSE.txt", "License (Overlay Asset)\n"))
    return zip_path


def open_path(path):
    """Reveal a file or folder in the OS file browser."""
    try:
        if os.name == "nt":
            if os.path.isfile(path):
                subprocess.Popen(["explorer", "/select,", os.path.normpath(path)], creationflags=_no_window_flags())
            else:
                os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path] if os.path.isfile(path) else ["open", path])
        else:
            target = path if os.path.isdir(path) else os.path.dirname(path)
            subprocess.Popen(["xdg-open", target], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def platform_label():
    return f"{platform.system()} {platform.release()}"
