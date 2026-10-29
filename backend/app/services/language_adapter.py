"""
ChronoVideo - Multi-Language Notes & Concept Adapter
Translates and culturally adapts technical lecture study notes into
Hinglish, Hindi (हिन्दी), Spanish, French, German, Japanese, Bengali, Tamil, etc.,
while preserving code, timestamps, and mathematical formulas.
Default language is always English.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .file_processor import get_groq_api_key

try:
    from groq import Groq
except ImportError:
    Groq = None

logger = logging.getLogger("chronovideo.lang")

# Supported language codes and descriptive prompts
LANGUAGE_DESCRIPTIONS = {
    "en": "English",
    "hinglish": "Hinglish (Natural conversational Hindi written in Roman/English script, mixing standard CS/technical English terms like Stack, Tree, Gradient, Learning Rate, Time Complexity)",
    "hi": "Hindi (हिन्दी - clean student-friendly Hindi in Devanagari script, preserving technical terms, variable names, and formulas in standard English/Latin)",
    "es": "Spanish (Español - natural academic Spanish for engineering students)",
    "fr": "French (Français - clear academic French for technical students)",
    "de": "German (Deutsch - precise academic German for university students)",
    "ja": "Japanese (日本語 - polite technical Japanese keeping code terms in English)",
    "bn": "Bengali (বাংলা - clean engineering Bengali keeping technical formulas in English)",
    "ta": "Tamil (தமிழ் - clear student-friendly Tamil keeping technical terms in English)",
    "te": "Telugu (తెలుగు - clear student-friendly Telugu keeping technical terms in English)"
}

# In-memory translation cache: key = f"{video_id}_{style}_{lang}"
_translation_cache: Dict[str, Dict[str, Any]] = {}


def adapt_notes_language(
    video_id: str,
    style: str,
    notes_data: Dict[str, Any],
    target_lang: str = "en"
) -> Dict[str, Any]:
    """
    Adapts notes into the requested language.
    If target_lang is 'en' or empty, returns notes_data as-is immediately.
    """
    lang = (target_lang or "en").lower().strip()
    
    # 1. Default English rule: if English or empty, return original immediately
    if lang == "en" or not lang:
        return notes_data

    # 2. Check cache
    cache_key = f"{video_id}_{style}_{lang}"
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    lang_desc = LANGUAGE_DESCRIPTIONS.get(lang, lang)

    # 3. Check Groq availability
    groq_key = get_groq_api_key()
    if not groq_key or not Groq:
        logger.warning("Groq not configured for translation, returning default English notes.")
        return notes_data

    # Prepare minimal payload for LLM (only textual fields to keep latency sub-second)
    raw_sections = notes_data.get("sections", [])
    translatable_sections = []
    for s in raw_sections:
        translatable_sections.append({
            "heading": s.get("heading", ""),
            "box": s.get("box", ""),
            "points": s.get("points", [])
        })

    payload_to_translate = {
        "summary": notes_data.get("summary", ""),
        "sections": translatable_sections
    }

    prompt = f"""You are a master bilingual technical tutor for university students.
Translate and naturally adapt the following study notes JSON into {lang_desc}.

STRICT REQUIREMENTS:
1. Preserve the JSON structure exactly. Return valid JSON only with keys: 'summary' and 'sections' (each with 'heading', 'box', 'points').
2. Keep all code, syntax, variable names, math symbols (e.g. O(N), alpha, grad C, LIFO, std::stack), and CS technical names in English.
3. Make the tone friendly, crisp, and intuitive for a student revising for an exam.
4. Output ONLY the raw JSON object. Do not wrap in markdown quotes or add conversational filler.

Input JSON:
{json.dumps(payload_to_translate, ensure_ascii=False)}
"""

    try:
        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        content = resp.choices[0].message.content.strip()
        parsed = json.loads(content)

        # Merge translated fields back into original sections (preserving time_str and start_sec)
        translated_sections_list = parsed.get("sections", [])
        merged_sections = []
        for i, original_sec in enumerate(raw_sections):
            new_sec = dict(original_sec)
            if i < len(translated_sections_list):
                t_sec = translated_sections_list[i]
                if t_sec.get("heading"):
                    new_sec["heading"] = t_sec["heading"]
                if "box" in t_sec:
                    new_sec["box"] = t_sec["box"]
                if t_sec.get("points"):
                    new_sec["points"] = t_sec["points"]
            merged_sections.append(new_sec)

        translated_summary = parsed.get("summary", notes_data.get("summary", ""))

        # Rebuild Markdown text in target language
        rebuilt_md = []
        rebuilt_md.append(f"# {notes_data.get('title', 'Lecture Notes')}\n")
        if translated_summary:
            rebuilt_md.append(f"> {translated_summary}\n")

        for s in merged_sections:
            t_str = s.get("time_str", "")
            time_prefix = f"[{t_str}] " if t_str else ""
            rebuilt_md.append(f"## {time_prefix}{s.get('heading', '')}\n")
            if s.get("box"):
                rebuilt_md.append(f"> **KEY CONCEPT:** {s.get('box')}\n")
            for p in s.get("points", []):
                rebuilt_md.append(f"- {p}")
            rebuilt_md.append("\n")

        code_snippet = notes_data.get("code_snippet", "")
        if code_snippet:
            rebuilt_md.append(f"## Implementation Code\n```cpp\n{code_snippet}\n```\n")

        complexity = notes_data.get("complexity", "")
        if complexity:
            rebuilt_md.append(f"## Complexity Analysis\n{complexity}\n")

        final_adapted_notes = {
            **notes_data,
            "language": lang,
            "summary": translated_summary,
            "sections": merged_sections,
            "markdown": "\n".join(rebuilt_md)
        }

        # Cache the result
        _translation_cache[cache_key] = final_adapted_notes
        return final_adapted_notes

    except Exception as e:
        logger.error(f"Error adapting notes to language '{lang}': {e}", exc_info=True)
        # Fallback to original English notes gracefully
        return notes_data
