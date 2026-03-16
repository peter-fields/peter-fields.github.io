"""
Generate study PDFs from markdown files.
- Smaller font overall
- Each top-level heading (##) starts a new page
- Extra whitespace between ideas for notes
Usage: python generate_pdfs.py
"""

import os
import re
from xhtml2pdf import pisa

STUDY_MARKDOWNS = os.path.join(os.path.dirname(__file__), "study_markdowns")
TENSOR_NOTATION = "/Users/pfields/Git/peter-fields.github.io/notebooks/tensor_notation/tensor_notation_settled.md"
OUT_DIR = os.path.join(os.path.dirname(__file__), "study_pdfs")

FILES = [
    (os.path.join(STUDY_MARKDOWNS, "memorize_sheet.md"),       "memorize_sheet.pdf"),
    (os.path.join(STUDY_MARKDOWNS, "reference_sheet.md"),      "reference_sheet.pdf"),
    (os.path.join(STUDY_MARKDOWNS, "test_structure.md"),       "test_structure.pdf"),
    (os.path.join(STUDY_MARKDOWNS, "advanced_topics.md"),      "advanced_topics.pdf"),
    (TENSOR_NOTATION,                                           "tensor_notation_settled.pdf"),
    (os.path.join(STUDY_MARKDOWNS, "python_numpy_gotchas.md"), "python_numpy_gotchas.pdf"),
]

CSS = """
<style>
  @page {
    margin: 2cm 3cm 2cm 1.8cm;  /* wider right margin for handwritten notes */
  }
  body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 9pt;
    line-height: 1.8;
    color: #111;
  }
  h1 {
    font-size: 13pt;
    margin-top: 0;
    margin-bottom: 6pt;
    border-bottom: 1px solid #999;
    padding-bottom: 3pt;
  }
  /* Each h2 section starts on a new page */
  h2 {
    font-size: 11pt;
    page-break-before: always;
    margin-top: 0;
    margin-bottom: 8pt;
    border-bottom: 1px solid #ccc;
    padding-bottom: 2pt;
  }
  /* First h2 on the page — don't force a break before it */
  h2:first-of-type {
    page-break-before: avoid;
  }
  h3 {
    font-size: 10pt;
    margin-top: 18pt;
    margin-bottom: 4pt;
  }
  h4 {
    font-size: 9pt;
    margin-top: 14pt;
    margin-bottom: 3pt;
    font-style: italic;
  }
  p {
    margin-top: 0;
    margin-bottom: 10pt;
  }
  ul, ol {
    margin-top: 0;
    margin-bottom: 10pt;
    padding-left: 18pt;
  }
  li {
    margin-bottom: 5pt;
  }
  pre {
    background: #fafafa;
    border: 1px solid #e8e8e8;
    border-radius: 3px;
    padding: 10pt 12pt;
    font-size: 8.5pt;
    line-height: 1.7;
    white-space: pre-wrap;
    word-wrap: break-word;
    margin-bottom: 14pt;
  }
  code {
    font-size: 8.5pt;
    background: #fafafa;
    padding: 1pt 2pt;
  }
  pre code {
    background: none;
    padding: 0;
    font-size: 8.5pt;
  }
  /* Note-taking whitespace: blank lines between logical blocks */
  hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 14pt 0;
  }
  blockquote {
    border-left: 3px solid #ccc;
    margin: 0 0 10pt 0;
    padding: 4pt 10pt;
    color: #444;
    font-size: 8.5pt;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8pt;
    margin-bottom: 12pt;
  }
  th {
    background: #eee;
    border: 1px solid #bbb;
    padding: 4pt 6pt;
    text-align: left;
  }
  td {
    border: 1px solid #ccc;
    padding: 4pt 6pt;
    vertical-align: top;
  }
</style>
"""


def md_to_html(md_text):
    """Minimal markdown → HTML converter (no external deps beyond stdlib)."""
    lines = md_text.split("\n")
    html_lines = []
    in_code = False
    code_lines = []
    in_ul = False
    in_ol = False

    def close_list():
        nonlocal in_ul, in_ol
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

    def inline(text):
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        # Inline code
        text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
        # Links
        text = re.sub(r'\[(.+?)\]\((.+?)\)', r'\1', text)
        return text

    for line in lines:
        # Fenced code blocks
        if line.startswith("```"):
            if in_code:
                def fmt_code_line(l):
                    # preserve leading indentation, use br for line breaks
                    indent = len(l) - len(l.lstrip(" "))
                    return "&nbsp;" * indent + l.lstrip(" ")
                escaped = "<br/>".join(fmt_code_line(l) for l in code_lines)
                html_lines.append(f"<pre><code>{escaped}</code></pre>")
                code_lines = []
                in_code = False
            else:
                close_list()
                in_code = True
            continue
        if in_code:
            code_lines.append(line.replace("<", "&lt;").replace(">", "&gt;"))
            continue

        # Headings
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        if m:
            close_list()
            level = len(m.group(1))
            text = inline(m.group(2))
            html_lines.append(f"<h{level}>{text}</h{level}>")
            continue

        # Horizontal rule
        if re.match(r'^[-*_]{3,}\s*$', line):
            close_list()
            html_lines.append("<hr/>")
            continue

        # Unordered list
        m = re.match(r'^[\*\-]\s+(.*)', line)
        if m:
            if not in_ul:
                close_list()
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{inline(m.group(1))}</li>")
            continue

        # Ordered list
        m = re.match(r'^\d+\.\s+(.*)', line)
        if m:
            if not in_ol:
                close_list()
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{inline(m.group(1))}</li>")
            continue

        # Blank line
        if line.strip() == "":
            close_list()
            html_lines.append("<p>&nbsp;</p>")  # extra whitespace for notes
            continue

        # Regular paragraph line
        close_list()
        html_lines.append(f"<p>{inline(line)}</p>")

    close_list()
    if in_code:
        html_lines.append("</pre>")

    return "\n".join(html_lines)


def convert(md_path, pdf_path):
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    body = md_to_html(md)
    html = f"<html><head><meta charset='utf-8'>{CSS}</head><body>{body}</body></html>"

    with open(pdf_path, "wb") as f:
        result = pisa.CreatePDF(html, dest=f)

    if result.err:
        print(f"  ERROR: {md_path}")
    else:
        print(f"  OK: {os.path.basename(pdf_path)}")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for md_path, pdf_name in FILES:
        pdf_path = os.path.join(OUT_DIR, pdf_name)
        print(f"Converting {os.path.basename(md_path)}...")
        convert(md_path, pdf_path)
    print("Done.")
