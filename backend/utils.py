import aiofiles
import urllib
import mistune
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Union


def slugify_output_label(label: str, default: str = "run", max_length: int = 48) -> str:
    """Normalize a query-like label into a safe, readable directory suffix."""
    normalized = str(label or "").strip()
    if not normalized:
        return default

    normalized = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", " ", normalized)
    normalized = normalized.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip(" .")
    if not normalized:
        return default

    return normalized[:max_length].rstrip(" .") or default


def create_output_session_dir(
    label: str,
    *,
    base_dir: Union[str, Path] = "outputs",
    timestamp: str | None = None,
) -> str:
    """Create a timestamped session directory for a single practical run."""
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    slug = slugify_output_label(label)
    root = Path(base_dir)
    candidate = root / f"{ts}_{slug}"
    suffix = 2
    while candidate.exists():
        candidate = root / f"{ts}_{slug}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate.as_posix()


def _resolve_output_file_path(filename: str, extension: str, output_dir: str | None = None) -> Path:
    """Resolve an output artifact path either in a session dir or the legacy root."""
    stem = (filename or "report").strip() or "report"
    if output_dir:
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir / f"{stem}{extension}"
    return Path("outputs") / f"{stem[:60]}{extension}"

async def write_to_file(filename: str, text: str) -> None:
    """Asynchronously write text to a file in UTF-8 encoding.

    Args:
        filename (str): The filename to write to.
        text (str): The text to write.
    """
    # Ensure text is a string
    if not isinstance(text, str):
        text = str(text)

    # Convert text to UTF-8, replacing any problematic characters
    text_utf8 = text.encode('utf-8', errors='replace').decode('utf-8')

    async with aiofiles.open(filename, "w", encoding='utf-8') as file:
        await file.write(text_utf8)

async def write_text_to_md(text: str, filename: str = "", output_dir: str | None = None) -> str:
    """Writes text to a Markdown file and returns the file path.

    Args:
        text (str): Text to write to the Markdown file.

    Returns:
        str: The file path of the generated Markdown file.
    """
    file_path = _resolve_output_file_path(filename, ".md", output_dir)
    await write_to_file(str(file_path), text)
    return urllib.parse.quote(file_path.as_posix())

def _preprocess_images_for_pdf(text: str) -> str:
    """Convert web image URLs to absolute file paths for PDF generation.
    
    Transforms /outputs/images/... URLs to absolute file:// paths that
    weasyprint can resolve.
    """
    import re
    
    base_path = os.path.abspath(".")
    
    # Pattern to find markdown images with /outputs/ URLs
    def replace_image_url(match):
        alt_text = match.group(1)
        url = match.group(2)
        
        # Convert /outputs/... to absolute path
        if url.startswith("/outputs/"):
            abs_path = os.path.join(base_path, url.lstrip("/"))
            return f"![{alt_text}]({abs_path})"
        return match.group(0)
    
    # Match ![alt text](/outputs/images/...)
    pattern = r'!\[([^\]]*)\]\((/outputs/[^)]+)\)'
    return re.sub(pattern, replace_image_url, text)


async def write_md_to_pdf(text: str, filename: str = "", output_dir: str | None = None) -> str:
    """Converts Markdown text to a PDF file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated PDF.
    """
    file_path = _resolve_output_file_path(filename, ".pdf", output_dir)

    try:
        # Resolve css path relative to this backend module to avoid
        # dependency on the current working directory.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        css_path = os.path.join(current_dir, "styles", "pdf_styles.css")
        
        # Preprocess image URLs for PDF compatibility
        processed_text = _preprocess_images_for_pdf(text)
        
        # Set base_url to current directory for resolving any remaining relative paths
        base_url = os.path.abspath(".")

        from md2pdf.core import md2pdf
        md2pdf(str(file_path),
               md_content=processed_text,
               # md_file_path=f"{file_path}.md",
               css_file_path=css_path,
               base_url=base_url)
        print(f"Report written to {file_path}")
    except Exception as e:
        print(f"Error in converting Markdown to PDF: {e}")
        return ""

    encoded_file_path = urllib.parse.quote(file_path.as_posix())
    return encoded_file_path

async def write_md_to_word(text: str, filename: str = "", output_dir: str | None = None) -> str:
    """Converts Markdown text to a DOCX file and returns the file path.

    Args:
        text (str): Markdown text to convert.

    Returns:
        str: The encoded file path of the generated DOCX.
    """
    file_path = _resolve_output_file_path(filename, ".docx", output_dir)

    try:
        from docx import Document
        from htmldocx import HtmlToDocx
        # Convert report markdown to HTML
        html = mistune.html(text)
        # Create a document object
        doc = Document()
        # Convert the html generated from the report to document format
        HtmlToDocx().add_html_to_document(html, doc)

        # Saving the docx document to file_path
        doc.save(str(file_path))

        print(f"Report written to {file_path}")

        encoded_file_path = urllib.parse.quote(file_path.as_posix())
        return encoded_file_path

    except Exception as e:
        print(f"Error in converting Markdown to DOCX: {e}")
        return ""
