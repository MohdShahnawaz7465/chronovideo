"""
ChronoVideo - API Router
Exposes REST endpoints for indexing, semantic search, chapter discovery, and notes generation.
"""

import os
import re
import uuid
import shutil
import tempfile
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from ..services.ingestion import extract_youtube_id, fetch_video_transcript, get_youtube_metadata
from ..services.ml_clustering import detect_topic_chapters
from ..services.vector_db import vector_store
from ..services.file_processor import (
    process_uploaded_pdf,
    process_uploaded_media,
    generate_pdf_study_notes,
    UPLOAD_DIR,
    PDF_EXTENSIONS,
    AUDIO_EXTENSIONS,
    VIDEO_EXTENSIONS
)
from ..services.rag_synthesizer import (
    synthesize_temporal_answer,
    generate_lecture_study_notes,
    generate_lecture_quiz,
    generate_lecture_flashcards,
    generate_video_suggestions
)
from ..services.pdf_exporter import (
    build_ruled_notebook_html,
    render_html_to_pdf
)
from ..services.language_adapter import (
    adapt_notes_language,
    LANGUAGE_DESCRIPTIONS
)

router = APIRouter(prefix="/api", tags=["ChronoVideo"])

# In-memory store for active video sessions
video_cache: Dict[str, Dict[str, Any]] = {}


@router.get("/health")
@router.get("/status")
async def health_check():
    """Health check endpoint for VS Code and monitoring."""
    return {"status": "ok", "service": "ChronoVideo API"}



class IndexRequest(BaseModel):
    url: str = Field(..., description="YouTube URL or 11-char Video ID")


class SearchRequest(BaseModel):
    video_id: str
    query: str


class NotesRequest(BaseModel):
    video_id: str
    style: str = "detailed"  # "detailed", "asis", "short"
    language: str = "en"     # Defaults to English if none selected


class ExportPdfBody(BaseModel):
    video_id: str
    style: str = "detailed"
    language: str = "en"     # Defaults to English
    title: Optional[str] = None
    instructor: Optional[str] = None


@router.post("/index")
async def index_lecture(req: IndexRequest):
    """Parses, segments, clusters, and indexes a technical lecture."""
    vid = extract_youtube_id(req.url)
    if not vid:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL or Video ID.")

    # Return cached session instantaneously (0ms round-trip)
    if vid in video_cache:
        cached = video_cache[vid]
        return {
            "success": True,
            "video_id": vid,
            "metadata": cached.get("metadata", {}),
            "indexed_chunks": len(cached.get("segments", [])),
            "chapters": cached.get("chapters", []),
            "segments": cached.get("segments", []),
            "suggestions": cached.get("suggestions", [])
        }

    # 1. Fetch metadata and temporal transcript chunks
    data = fetch_video_transcript(vid)
    segments = data.get("segments", [])
    metadata = data.get("metadata", {})

    # 2. Unsupervised ML: Topic Boundary Detection (Chapters)
    chapters = detect_topic_chapters(segments)

    # 3. Vector Database Indexing
    indexed_count = vector_store.index_video_chunks(vid, segments)

    # 4. Generate intelligent query suggestions for autocomplete
    suggestions = generate_video_suggestions(metadata, chapters, segments)

    # Cache session
    video_cache[vid] = {
        "metadata": metadata,
        "segments": segments,
        "chapters": chapters,
        "suggestions": suggestions,
        "source": data.get("source", "curated")
    }

    return {
        "success": True,
        "video_id": vid,
        "metadata": metadata,
        "indexed_chunks": indexed_count,
        "chapters": chapters,
        "segments": segments,
        "suggestions": suggestions
    }


@router.post("/upload-file")
async def upload_lecture_file(file: UploadFile = File(...)):
    """Uploads and indexes an offline lecture file (PDF slides, MP3/WAV audio, MP4/MKV video)."""
    filename = file.filename or "lecture_file"
    ext = os.path.splitext(filename)[1].lower()
    allowed = PDF_EXTENSIONS | AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Please upload PDF slides (.pdf) or audio/video (.mp3, .wav, .m4a, .mp4, .mkv, .webm)."
        )

    file_id = f"local_{uuid.uuid4().hex[:10]}"
    saved_name = f"{file_id}{ext}"
    saved_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(saved_path, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    try:
        if ext in PDF_EXTENSIONS:
            result = process_uploaded_pdf(saved_path, filename)
        else:
            result = process_uploaded_media(saved_path, filename, ext)

        metadata = result["metadata"]
        metadata["id"] = file_id
        metadata["file_url"] = f"/api/files/{saved_name}"

        segments = result["segments"]
        chapters = result["chapters"]
        suggestions = result["suggestions"]

        # Index in vector store for high-speed semantic search & RAG
        indexed_count = vector_store.index_video_chunks(file_id, segments)

        # Cache session in memory
        video_cache[file_id] = {
            "metadata": metadata,
            "segments": segments,
            "raw_segments": result.get("raw_segments", []),
            "chapters": chapters,
            "suggestions": suggestions,
            "saved_path": saved_path
        }

        return {
            "success": True,
            "video_id": file_id,
            "metadata": metadata,
            "indexed_chunks": indexed_count,
            "chapters": chapters,
            "segments": segments,
            "suggestions": suggestions,
            "file_url": metadata["file_url"]
        }
    except Exception as e:
        if os.path.exists(saved_path):
            try:
                os.remove(saved_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.get("/files/{file_name}")
async def serve_uploaded_file(file_name: str):
    """Serves an uploaded local lecture file for HTML5 playback or PDF viewer."""
    file_path = os.path.join(UPLOAD_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


@router.get("/suggestions")
async def get_suggestions(video_id: str = Query(...), q: str = Query(default="")):
    """Returns autocomplete suggestions for the given video and partial search query."""
    cache = video_cache.get(video_id)
    if not cache:
        meta = get_youtube_metadata(video_id)
        suggestions = generate_video_suggestions(meta, [], [])
    else:
        suggestions = cache.get("suggestions", [])
        if not suggestions:
            suggestions = generate_video_suggestions(
                cache.get("metadata", {}),
                cache.get("chapters", []),
                cache.get("segments", [])
            )
            cache["suggestions"] = suggestions

    q_clean = q.strip().lower()
    if not q_clean:
        return {"suggestions": suggestions[:8]}

    # Priority 1: Prefix match
    prefix_matches = [s for s in suggestions if s.lower().startswith(q_clean)]
    # Priority 2: Substring match
    sub_matches = [s for s in suggestions if q_clean in s.lower() and s not in prefix_matches]
    # Priority 3: Word match
    words = [w for w in q_clean.split() if len(w) > 2]
    word_matches = []
    if words:
        for s in suggestions:
            if s not in prefix_matches and s not in sub_matches:
                if any(w in s.lower() for w in words):
                    word_matches.append(s)

    combined = (prefix_matches + sub_matches + word_matches)[:10]
    return {"suggestions": combined}


@router.post("/search")
async def search_lecture(req: SearchRequest):
    """Semantic Temporal Search & RAG:
    Finds the exact timestamp and synthesizes a citation-backed explanation."""
    if not req.video_id:
        raise HTTPException(status_code=400, detail="Missing video_id.")

    # If video wasn't indexed yet, index it first
    if req.video_id not in video_cache:
        data = fetch_video_transcript(req.video_id)
        segments = data.get("segments", [])
        vector_store.index_video_chunks(req.video_id, segments)
        chapters = detect_topic_chapters(segments)
        video_cache[req.video_id] = {
            "metadata": data.get("metadata", {}),
            "segments": segments,
            "chapters": chapters
        }

    res = synthesize_temporal_answer(req.video_id, req.query)
    return res


@router.post("/generate-notes")
async def generate_notes(req: NotesRequest):
    """Generates structured notes (detailed, as-is, or short) with handwritten paper sections and markdown."""
    info = video_cache.get(req.video_id)
    if not info:
        data = fetch_video_transcript(req.video_id)
        info = {
            "metadata": data.get("metadata", {}),
            "segments": data.get("segments", []),
            "chapters": detect_topic_chapters(data.get("segments", []))
        }
        video_cache[req.video_id] = info

    title = info["metadata"].get("title", f"Lecture {req.video_id}")
    
    # If this is an uploaded PDF slide deck, use PDF-tailored slide notes synthesizer
    if info.get("metadata", {}).get("type") == "pdf":
        raw_segs = info.get("raw_segments", info.get("segments", []))
        notes_data = generate_pdf_study_notes(title=title, raw_segments=raw_segs, style=req.style)
    else:
        notes_data = generate_lecture_study_notes(
            video_id=req.video_id,
            title=title,
            segments=info["segments"],
            chapters=info["chapters"],
            style=req.style
        )

    target_lang = (req.language or "en").lower().strip()
    if target_lang and target_lang != "en":
        notes_data = adapt_notes_language(
            video_id=req.video_id,
            style=req.style,
            notes_data=notes_data,
            target_lang=target_lang
        )

    return {
        "success": True,
        "video_id": req.video_id,
        "title": title,
        "style": req.style,
        "language": target_lang,
        **notes_data
    }


def generate_ruled_pdf_file(
    video_id: str,
    style: str = "detailed",
    language: str = "en",
    override_title: Optional[str] = None,
    override_instructor: Optional[str] = None
) -> tuple[str, str]:
    """Compiles lecture study notes into a physical-styled ruled notebook PDF."""
    info = video_cache.get(video_id)
    if not info:
        data = fetch_video_transcript(video_id)
        info = {
            "metadata": data.get("metadata", {}),
            "segments": data.get("segments", []),
            "chapters": detect_topic_chapters(data.get("segments", []))
        }
        video_cache[video_id] = info

    title = override_title or info.get("metadata", {}).get("title", f"Lecture {video_id}")
    instructor = override_instructor or info.get("metadata", {}).get("author", "Technical Course")

    if info.get("metadata", {}).get("type") == "pdf":
        raw_segs = info.get("raw_segments", info.get("segments", []))
        notes_data = generate_pdf_study_notes(title=title, raw_segments=raw_segs, style=style)
    else:
        notes_data = generate_lecture_study_notes(
            video_id=video_id,
            title=title,
            segments=info["segments"],
            chapters=info["chapters"],
            style=style
        )

    target_lang = (language or "en").lower().strip()
    if target_lang and target_lang != "en":
        notes_data = adapt_notes_language(
            video_id=video_id,
            style=style,
            notes_data=notes_data,
            target_lang=target_lang
        )

    style_label = "Teacher-Verbatim" if style == "asis" else ("Quick Cheat Sheet" if style == "short" else "Detailed Notebook")
    if target_lang and target_lang != "en":
        style_label += f" ({target_lang.capitalize()})"

    html_content = build_ruled_notebook_html(
        title=title,
        instructor=instructor,
        style_label=style_label,
        summary=notes_data.get("summary", ""),
        sections=notes_data.get("sections", []),
        code_snippet=notes_data.get("code_snippet", ""),
        complexity=notes_data.get("complexity", "")
    )

    clean_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:35].strip('_') or "Lecture"
    lang_suffix = f"_{target_lang}" if target_lang != "en" else ""
    pdf_filename = f"ChronoVideo_{clean_name}_{style}{lang_suffix}_Notes.pdf"

    tmp_pdf = tempfile.NamedTemporaryFile("wb", suffix=".pdf", delete=False)
    tmp_pdf_path = tmp_pdf.name
    tmp_pdf.close()

    success = render_html_to_pdf(html_content, tmp_pdf_path)
    if not success or not os.path.exists(tmp_pdf_path) or os.path.getsize(tmp_pdf_path) == 0:
        raise HTTPException(status_code=500, detail="Failed to compile ruled PDF document")

    return tmp_pdf_path, pdf_filename


@router.get("/export-notes-pdf")
async def export_notes_pdf_get(
    video_id: str = Query(..., description="Video ID or resource key"),
    style: str = Query("detailed", description="Note style: detailed, asis, short"),
    language: str = Query("en", description="Target language (default: en)")
):
    """Generates and downloads a real physical-styled ruled notebook PDF."""
    pdf_path, filename = generate_ruled_pdf_file(video_id=video_id, style=style, language=language)
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.post("/export-notes-pdf")
async def export_notes_pdf_post(req: ExportPdfBody):
    """POST endpoint to generate and download ruled notebook PDF."""
    pdf_path, filename = generate_ruled_pdf_file(
        video_id=req.video_id,
        style=req.style,
        language=req.language or "en",
        override_title=req.title,
        override_instructor=req.instructor
    )
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.get("/languages")
async def get_supported_languages():
    """Returns list of supported note languages with English as default."""
    return [
        {"code": "en", "name": "English", "native": "English", "flag": "🇬🇧", "default": True},
        {"code": "hinglish", "name": "Hinglish", "native": "Hinglish", "flag": "🇮🇳"},
        {"code": "hi", "name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳"},
        {"code": "es", "name": "Spanish", "native": "Español", "flag": "🇪🇸"},
        {"code": "fr", "name": "French", "native": "Français", "flag": "🇫🇷"},
        {"code": "de", "name": "German", "native": "Deutsch", "flag": "🇩🇪"},
        {"code": "ja", "name": "Japanese", "native": "日本語", "flag": "🇯🇵"},
        {"code": "bn", "name": "Bengali", "native": "বাংলা", "flag": "🇮🇳"},
        {"code": "ta", "name": "Tamil", "native": "தமிழ்", "flag": "🇮🇳"},
        {"code": "te", "name": "Telugu", "native": "తెలుగు", "flag": "🇮🇳"}
    ]


@router.post("/generate-quiz")
async def get_lecture_quiz(req: NotesRequest):
    """Generates an interactive multiple-choice quiz from the lecture content."""
    info = video_cache.get(req.video_id)
    if not info:
        data = fetch_video_transcript(req.video_id)
        info = {
            "metadata": data.get("metadata", {}),
            "segments": data.get("segments", []),
            "chapters": detect_topic_chapters(data.get("segments", []))
        }
        video_cache[req.video_id] = info

    title = info["metadata"].get("title", f"Lecture {req.video_id}")
    quiz_items = generate_lecture_quiz(
        video_id=req.video_id,
        title=title,
        segments=info["segments"],
        chapters=info["chapters"]
    )
    return {
        "success": True,
        "video_id": req.video_id,
        "title": title,
        "quiz": quiz_items
    }


@router.post("/generate-flashcards")
async def get_lecture_flashcards(req: NotesRequest):
    """Generates concept flashcards for active recall study."""
    info = video_cache.get(req.video_id)
    if not info:
        data = fetch_video_transcript(req.video_id)
        info = {
            "metadata": data.get("metadata", {}),
            "segments": data.get("segments", []),
            "chapters": detect_topic_chapters(data.get("segments", []))
        }
        video_cache[req.video_id] = info

    title = info["metadata"].get("title", f"Lecture {req.video_id}")
    cards = generate_lecture_flashcards(
        video_id=req.video_id,
        title=title,
        segments=info["segments"],
        chapters=info["chapters"]
    )
    return {
        "success": True,
        "video_id": req.video_id,
        "title": title,
        "flashcards": cards
    }


@router.get("/sample-lectures")
async def get_sample_lectures():
    """Returns curated featured lectures for instant 1-click evaluation."""
    return [
        {
            "id": "Bfqd8BsPVuw",
            "title": "Iterative Preorder Traversal in Binary Tree | C++ | Java",
            "instructor": "take U forward (Striver)",
            "duration": "8 min",
            "topics": ["Binary Trees", "Stacks", "Preorder Traversal", "LIFO Order"]
        },
        {
            "id": "IHZwWFHWa-w",
            "title": "3Blue1Brown: Gradient Descent, How Neural Networks Learn",
            "instructor": "3Blue1Brown",
            "duration": "21 min",
            "topics": ["Cost Functions", "Gradient Vectors", "Negative Gradient", "Learning Rate"]
        },
        {
            "id": "zjkBMFhNj_g",
            "title": "Andrej Karpathy: Intro to Large Language Models",
            "instructor": "Andrej Karpathy",
            "duration": "60 min",
            "topics": ["Pre-training", "Fine-tuning", "Tokenization", "Hallucinations"]
        },
        {
            "id": "ZK3O402wf1c",
            "title": "MIT 18.06: The Geometry of Linear Equations",
            "instructor": "Gilbert Strang (MIT)",
            "duration": "39 min",
            "topics": ["Row Picture", "Column Picture", "Matrix Equations", "Linear Combinations"]
        }
    ]
