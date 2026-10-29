"""
ChronoVideo - Local File Processor Service
Handles local Audio (.mp3, .wav, .m4a), Video (.mp4, .mkv, .webm), and PDF Slide uploads.
Integrates PyMuPDF (fitz) for slides and Groq Whisper + ffmpeg for speech transcription.
"""

import os
import re
import sys
import glob
import json
import shutil
import tempfile
import subprocess
from typing import List, Dict, Any, Tuple, Optional

# Link to existing site-packages in LectureLens venv if needed
VENV_PKGS = r"C:\Users\Admin\Downloads\LectureLens_skeleton\lecturelens\backend\venv\Lib\site-packages"
if os.path.exists(VENV_PKGS) and VENV_PKGS not in sys.path:
    sys.path.append(VENV_PKGS)

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

try:
    from groq import Groq
except ImportError:
    Groq = None

from backend.app.services.ingestion import format_seconds, group_into_temporal_chunks
from backend.app.services.ml_clustering import detect_topic_chapters

# Resolve Groq API Key
def get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "")
    if key and not key.startswith("your_"):
        return key
    
    # Try reading from LectureLens .env
    env_candidate = r"C:\Users\Admin\Downloads\LectureLens_skeleton\lecturelens\backend\.env"
    if os.path.exists(env_candidate):
        try:
            with open(env_candidate, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("GROQ_API_KEY="):
                        val = line.strip().split("=", 1)[1].strip()
                        if val and not val.startswith("your_"):
                            return val
        except Exception:
            pass
    return key


UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
PDF_EXTENSIONS = {".pdf"}


def get_ffmpeg_path() -> Optional[str]:
    """Returns path to ffmpeg binary."""
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg
    if imageio_ffmpeg:
        try:
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass
    return None


def process_uploaded_pdf(file_path: str, original_filename: str) -> Dict[str, Any]:
    """Processes an uploaded PDF lecture slide file using PyMuPDF (fitz)."""
    if not fitz:
        raise RuntimeError("PyMuPDF (fitz) is not available.")

    doc = fitz.open(file_path)
    total_pages = len(doc)
    if total_pages == 0:
        raise ValueError("Uploaded PDF document is empty.")

    raw_segments = []
    chunks = []
    chapters = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        text = page.get_text().strip()
        page_num = page_idx + 1

        if not text:
            text = f"[Slide Page {page_num}: Diagram / Code Figure]"

        raw_segments.append({
            "start": float(page_num),
            "duration": 1.0,
            "text": text,
            "page": page_num
        })

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        first_line = lines[0] if lines else f"Page {page_num}"
        if len(first_line) > 55:
            first_line = first_line[:52] + "..."

        time_label = f"p.{page_num}"
        chunks.append({
            "chunk_id": f"chunk_p{page_num:03d}",
            "start_sec": float(page_num),
            "end_sec": float(page_num),
            "time_str": time_label,
            "end_time_str": time_label,
            "text": text,
            "page": page_num
        })

        # Slide chapter detection
        chapters.append({
            "id": f"ch_p{page_num}",
            "title": first_line,
            "start_sec": float(page_num),
            "time_str": time_label
        })

    title = os.path.splitext(original_filename)[0].replace("_", " ").title()
    metadata = {
        "title": title,
        "author": "Lecture Slides (PDF)",
        "thumbnail_url": "",
        "duration": f"{total_pages} slides",
        "type": "pdf",
        "total_pages": total_pages,
        "filename": original_filename
    }

    # Generate suggestions from slide headings
    suggestions = []
    for ch in chapters[:8]:
        t = ch.get("title", "")
        if t and not t.lower().startswith("page "):
            suggestions.append(t)

    return {
        "metadata": metadata,
        "segments": chunks,
        "raw_segments": raw_segments,
        "chapters": chapters,
        "suggestions": suggestions
    }


def process_uploaded_media(file_path: str, original_filename: str, ext: str) -> Dict[str, Any]:
    """Transcribes audio/video lecture using Groq Whisper + ffmpeg."""
    ffmpeg_exe = get_ffmpeg_path()
    if not ffmpeg_exe:
        raise RuntimeError("ffmpeg not found for audio extraction.")

    groq_key = get_groq_api_key()
    if not groq_key or not Groq:
        raise RuntimeError("Groq API key not configured for Whisper transcription.")

    client = Groq(api_key=groq_key)
    temp_dir = tempfile.mkdtemp(prefix="chrono_audio_")

    try:
        # Extract audio as 16kHz mono mp3
        converted_audio = os.path.join(temp_dir, "lecture_audio.mp3")
        cmd = [
            ffmpeg_exe, "-y", "-i", file_path,
            "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
            converted_audio
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(converted_audio):
            raise RuntimeError(f"ffmpeg audio conversion failed: {res.stderr[:200]}")

        # Transcribe with Groq Whisper
        with open(converted_audio, "rb") as fh:
            whisper_res = client.audio.transcriptions.create(
                file=(os.path.basename(converted_audio), fh.read()),
                model=os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"),
                response_format="verbose_json"
            )

        raw_segments = []
        seg_list = getattr(whisper_res, "segments", None)
        if seg_list is None and isinstance(whisper_res, dict):
            seg_list = whisper_res.get("segments", [])

        if seg_list:
            for s in seg_list:
                s_dict = s if isinstance(s, dict) else s.__dict__
                start = float(s_dict.get("start", 0.0))
                end = float(s_dict.get("end", start + 3.0))
                txt = s_dict.get("text", "").strip()
                if txt:
                    raw_segments.append({
                        "start": round(start, 2),
                        "duration": round(max(0.5, end - start), 2),
                        "text": txt
                    })
        else:
            full_txt = getattr(whisper_res, "text", "") or ""
            raw_segments.append({
                "start": 0.0,
                "duration": 60.0,
                "text": full_txt.strip() or "Lecture audio recorded."
            })

        # Calculate total duration
        total_duration = raw_segments[-1]["start"] + raw_segments[-1]["duration"] if raw_segments else 60.0

        # Group into structured temporal chunks for search & transcript
        chunks = group_into_temporal_chunks(raw_segments, target_duration=40.0)

        # Detect topic chapters
        chapters = detect_topic_chapters(raw_segments)

        title = os.path.splitext(original_filename)[0].replace("_", " ").title()
        metadata = {
            "title": title,
            "author": "Local Recorded Lecture",
            "thumbnail_url": "",
            "duration": format_seconds(total_duration),
            "type": "media",
            "media_type": "video" if ext in VIDEO_EXTENSIONS else "audio",
            "filename": original_filename,
            "total_seconds": total_duration
        }

        # Generate search suggestions
        suggestions = [ch.get("title", "") for ch in chapters if ch.get("title")][:8]

        return {
            "metadata": metadata,
            "segments": chunks,
            "raw_segments": raw_segments,
            "chapters": chapters,
            "suggestions": suggestions
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def generate_pdf_study_notes(title: str, raw_segments: List[Dict[str, Any]], style: str = "detailed") -> Dict[str, Any]:
    """Generates structured 3-mode notes from PDF lecture slides."""
    sections = []
    
    # Process slides into thematic sections
    for i, seg in enumerate(raw_segments):
        page_num = seg.get("page", i + 1)
        text = seg.get("text", "").strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        
        slide_title = lines[0] if lines else f"Slide Page {page_num}"
        body_lines = lines[1:] if len(lines) > 1 else lines
        
        box_text = ""
        points = []
        
        for line in body_lines:
            if not box_text and any(k in line.lower() for k in ["definition", "rule", "formula", "theorem", "key", "important"]):
                box_text = line
            else:
                # Add highlights to bold terms
                formatted = re.sub(r'([A-Z][a-zA-Z0-9_\-\+]{2,})', r'**\1**', line)
                points.append(formatted)
        
        if not box_text and body_lines:
            box_text = f"Key takeaway from slide {page_num}: {body_lines[0]}"
            
        if not points:
            points = [f"Reference slide content on page {page_num}."]

        sections.append({
            "heading": f"{slide_title}",
            "box": box_text,
            "start": float(page_num),
            "time_str": f"p.{page_num}",
            "points": points[:6]
        })

    # Filter or condense according to style
    if style == "short":
        summary = f"Quick 1-page revision cheat sheet extracted from {len(raw_segments)} lecture slides."
        sections = sections[:4]
        for sec in sections:
            sec["points"] = sec["points"][:3]
    elif style == "asis":
        summary = f"As-presented slide sequence preserving professor's exact slide hierarchy and layout."
    else:  # detailed
        summary = f"Comprehensive class study notes synthesizing {len(raw_segments)} lecture slides with formulas and rules."

    # Generate markdown equivalent
    md_lines = [
        f"# {title}",
        f"*{summary}*\n",
        f"**Source**: Uploaded Lecture Slides ({len(raw_segments)} pages)\n",
        "---",
        "## Core Concepts & Slide Breakdown\n"
    ]

    for sec in sections:
        md_lines.append(f"### [{sec['time_str']}] {sec['heading']}")
        if sec.get("box"):
            md_lines.append(f"> **KEY RULE**: {sec['box']}\n")
        for pt in sec.get("points", []):
            md_lines.append(f"- {pt}")
        md_lines.append("")

    return {
        "summary": summary,
        "sections": sections,
        "markdown": "\n".join(md_lines)
    }
