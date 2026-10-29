"""
ChronoVideo - Ruled Notebook PDF Exporter
Generates authentic, student-ready ruled notebook PDFs with handwritten-style typography,
binder punch rings, timestamp badges, callout boxes, and print-perfect A4 pagination.
"""

import os
import re
import html
import shutil
import tempfile
import subprocess
from datetime import datetime
from typing import Dict, Any, List, Optional

# Potential browser paths on Windows and Linux
BROWSER_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "google-chrome",
    "chrome",
    "chromium",
    "msedge"
]

def find_headless_browser() -> Optional[str]:
    """Finds available Chrome or Edge binary for headless PDF rendering."""
    for path in BROWSER_CANDIDATES:
        if os.path.isabs(path) and os.path.exists(path):
            return path
        found = shutil.which(path)
        if found:
            return found
    return None


def format_inline_markdown(text: str) -> str:
    """Formats bold, italic, and code within bullet points and callouts."""
    if not text:
        return ""
    escaped = html.escape(str(text))
    # Bold **text**
    escaped = re.sub(r'\*\*(.*?)\*\*', r'<strong style="color:#0f172a; font-weight:700;">\1</strong>', escaped)
    # Italic *text*
    escaped = re.sub(r'\*(.*?)\*', r'<em>\1</em>', escaped)
    # Inline code `code`
    escaped = re.sub(
        r'`([^`]+)`',
        r'<code style="background:#f1f5f9; color:#ea580c; padding:1px 5px; border-radius:3px; font-family:\'JetBrains Mono\', monospace; font-size:12px;">\1</code>',
        escaped
    )
    return escaped


def build_ruled_notebook_html(
    title: str,
    instructor: str = "Technical Course",
    style_label: str = "Detailed Notebook",
    summary: str = "",
    sections: Optional[List[Dict[str, Any]]] = None,
    code_snippet: str = "",
    complexity: str = ""
) -> str:
    """Builds complete printable HTML styled like a physical ruled spiral notebook."""
    sections = sections or []
    date_str = datetime.now().strftime("%d %b %Y")

    sections_html = []
    for idx, s in enumerate(sections):
        t_badge = s.get("time_str", "00:00")
        heading = s.get("heading", f"Topic {idx + 1}")
        box = s.get("box", "")
        points = s.get("points", [])

        box_html = ""
        if box:
            formatted_box = format_inline_markdown(box)
            box_html = f'<div class="callout-box"><strong>KEY CONCEPT:</strong> {formatted_box}</div>'

        li_items = []
        for p in points:
            p_str = str(p).strip()
            is_sub = p_str.startswith("↳") or p_str.startswith("->")
            clean = p_str.lstrip("↳-> ").strip()
            formatted_point = format_inline_markdown(clean)
            if is_sub:
                li_items.append(f'<li class="sub-point"><span class="sym">↳</span><span>{formatted_point}</span></li>')
            else:
                li_items.append(f'<li><span class="sym">•</span><span>{formatted_point}</span></li>')

        points_html = "\n".join(li_items)

        sections_html.append(f"""
        <div class="note-section">
            <div class="section-title-row">
                <span class="timestamp-badge">[{html.escape(t_badge)}]</span>
                <span class="section-heading">{html.escape(heading)}</span>
            </div>
            {box_html}
            <ul class="points-list">
                {points_html}
            </ul>
        </div>
        """)

    all_sections = "\n".join(sections_html)

    code_html = ""
    if code_snippet:
        escaped_code = html.escape(code_snippet.strip())
        code_html = f"""
        <div class="note-section code-section">
            <div class="section-title-row">
                <span class="section-heading" style="color:#b91c1c; text-decoration-color:#ea580c;">💻 Implementation Code</span>
            </div>
            <pre class="code-box"><code>{escaped_code}</code></pre>
        </div>
        """

    comp_html = ""
    if complexity:
        formatted_comp = format_inline_markdown(complexity).replace("\n", "<br>")
        comp_html = f"""
        <div class="note-section">
            <div class="callout-box" style="background:#fef3c7; border-left:3.5px solid #d97706; color:#92400e;">
                <strong>⚡ COMPLEXITY ANALYSIS:</strong><br>{formatted_comp}
            </div>
        </div>
        """

    formatted_summary = format_inline_markdown(summary)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{html.escape(title)} — ChronoVideo Ruled Notes</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Kalam:wght@400;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  @page {{
    size: A4 portrait;
    margin: 10mm 12mm 10mm 12mm;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  body {{
    background-color: #fffef9;
    color: #1e3a8a;
    font-family: 'Kalam', 'Segoe Print', cursive, sans-serif;
    font-size: 15px;
    line-height: 29px;
    -webkit-print-color-adjust: exact !important;
    print-color-adjust: exact !important;
    padding: 0;
  }}
  .notebook-page {{
    position: relative;
    background-color: #fffef9;
    background-image: 
      linear-gradient(to right, transparent 58px, #f87171 58px, #f87171 60px, transparent 60px),
      repeating-linear-gradient(to bottom, transparent 0, transparent 28px, #dbeafe 28px, #dbeafe 29px);
    background-position: 0 0;
    padding: 12px 18px 24px 76px;
    min-height: 100vh;
  }}
  
  /* Spiral Binder Holes */
  .binder-hole {{
    position: absolute;
    left: 20px;
    width: 17px;
    height: 17px;
    border-radius: 50%;
    background: #e2e8f0;
    border: 1.5px solid #cbd5e1;
    box-shadow: inset 0 2px 3px rgba(0,0,0,0.15);
  }}
  .hole-1 {{ top: 40px; }}
  .hole-2 {{ top: 25%; }}
  .hole-3 {{ top: 50%; }}
  .hole-4 {{ top: 75%; }}
  .hole-5 {{ bottom: 40px; }}

  /* Notebook Header Banner */
  .notebook-header {{
    border-bottom: 2.5px solid #2563eb;
    padding-bottom: 8px;
    margin-bottom: 16px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    font-family: 'Inter', -apple-system, sans-serif;
  }}
  .brand-tag {{
    font-size: 11px;
    font-weight: 700;
    color: #ea580c;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 5px;
  }}
  .doc-title {{
    font-size: 19px;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.25;
    font-family: 'Inter', -apple-system, sans-serif;
    max-width: 500px;
  }}
  .meta-badges {{
    text-align: right;
    font-size: 11.5px;
    color: #64748b;
  }}
  .mode-pill {{
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    border: 1px solid #bae6fd;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 11px;
    margin-bottom: 4px;
  }}

  /* Summary Text */
  .summary-text {{
    font-family: 'Kalam', cursive;
    font-size: 14.5px;
    color: #1e40af;
    font-style: italic;
    margin-bottom: 16px;
    line-height: 27px;
  }}

  /* Section Styling */
  .note-section {{
    margin-bottom: 18px;
    break-inside: avoid;
    page-break-inside: avoid;
  }}
  .section-title-row {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
    break-after: avoid;
  }}
  .timestamp-badge {{
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 11.5px;
    font-weight: 600;
    background: #fef08a;
    border: 1px solid #facc15;
    color: #854d0e;
    padding: 1px 7px;
    border-radius: 4px;
    line-height: 18px;
    display: inline-block;
    flex-shrink: 0;
  }}
  .section-heading {{
    font-family: 'Kalam', cursive;
    font-weight: 700;
    font-size: 18px;
    color: #1e3a8a;
    line-height: 28px;
    text-decoration: underline;
    text-decoration-color: #f87171;
    text-decoration-thickness: 1.5px;
    text-underline-offset: 4px;
  }}
  
  /* Callout box */
  .callout-box {{
    background: #eff6ff;
    border-left: 3.5px solid #3b82f6;
    padding: 6px 12px;
    margin: 6px 0;
    border-radius: 4px;
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 12.5px;
    line-height: 20px;
    color: #1e3a8a;
    break-inside: avoid;
  }}

  /* Points List */
  .points-list {{
    list-style: none;
    padding: 0;
    margin: 4px 0 8px 0;
  }}
  .points-list li {{
    display: flex;
    gap: 8px;
    align-items: flex-start;
    line-height: 28px;
    font-size: 15px;
    color: #172554;
  }}
  .points-list li.sub-point {{
    padding-left: 18px;
    color: #2563eb;
    font-size: 14px;
  }}
  .sym {{
    color: #ef4444;
    font-weight: 700;
    flex-shrink: 0;
    user-select: none;
  }}

  /* Code Section */
  .code-box {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 3.5px solid #ea580c;
    border-radius: 6px;
    padding: 10px 14px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 11.5px;
    line-height: 18px;
    color: #0f172a;
    white-space: pre-wrap;
    margin: 6px 0;
  }}

  /* Printable Footer */
  .footer-stamp {{
    margin-top: 26px;
    padding-top: 8px;
    border-top: 1px dashed #cbd5e1;
    display: flex;
    justify-content: space-between;
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 10.5px;
    color: #94a3b8;
    break-inside: avoid;
  }}
</style>
</head>
<body>
  <div class="notebook-page">
    <div class="binder-hole hole-1"></div>
    <div class="binder-hole hole-2"></div>
    <div class="binder-hole hole-3"></div>
    <div class="binder-hole hole-4"></div>
    <div class="binder-hole hole-5"></div>

    <header class="notebook-header">
      <div>
        <div class="brand-tag">
          <span>●</span> ChronoVideo Lecture Study Notes
        </div>
        <h1 class="doc-title">{html.escape(title)}</h1>
      </div>
      <div class="meta-badges">
        <div><span class="mode-pill">{html.escape(style_label)}</span></div>
        <div><strong>Source:</strong> {html.escape(instructor)}</div>
        <div><strong>Exported:</strong> {date_str}</div>
      </div>
    </header>

    {f'<p class="summary-text">{formatted_summary}</p>' if summary else ''}

    {all_sections}

    {code_html}

    {comp_html}

    <footer class="footer-stamp">
      <span>Generated by ChronoVideo — Video Intelligence & Ruled Study Notes</span>
      <span>A4 Ruled Notebook Edition • Print & iPad Ready</span>
    </footer>
  </div>
</body>
</html>
"""
    return html_content


def render_html_to_pdf(html_content: str, output_pdf_path: str) -> bool:
    """Invokes headless Chrome or Edge to compile HTML to vector PDF."""
    browser_bin = find_headless_browser()
    if not browser_bin:
        raise RuntimeError("No headless Chrome or Edge browser found on host system.")

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False, encoding="utf-8") as tmp_html:
        tmp_html.write(html_content)
        tmp_html_path = tmp_html.name

    try:
        cmd = [
            browser_bin,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--print-to-pdf={output_pdf_path}",
            "--no-pdf-header-footer",
            tmp_html_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=25)
        if result.returncode != 0 or not os.path.exists(output_pdf_path) or os.path.getsize(output_pdf_path) == 0:
            return False
        return True
    finally:
        if os.path.exists(tmp_html_path):
            try:
                os.remove(tmp_html_path)
            except Exception:
                pass
