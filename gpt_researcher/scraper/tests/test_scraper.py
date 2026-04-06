"""
Unit tests for the web scraper layer.

Failure modes observed in 2026-04-06_193235 run logs:
  - "Error! : HTTPSConnectionPool... Read timed out" (BeautifulSoupScraper)
  - "Content too short or empty for <url>" (Scraper.extract_data_from_url)
  - PDF URLs returning empty content (PyMuPDFScraper timeout / HTTP error)

These tests verify:
  1. Scraper.get_scraper()         — URL-based backend routing
  2. Scraper.extract_data_from_url() — short content, exception, good content
  3. Scraper.run()                 — None filtering
  4. BeautifulSoupScraper.scrape() — timeout / network error / good HTML
  5. PyMuPDFScraper.scrape()       — timeout / HTTP error / empty doc / success
  6. PyMuPDFScraper.is_url()
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import requests

from gpt_researcher.scraper.scraper import Scraper
from gpt_researcher.scraper.beautiful_soup.beautiful_soup import BeautifulSoupScraper
from gpt_researcher.scraper.pymupdf.pymupdf import PyMuPDFScraper


# ── Helpers ──────────────────────────────────────────────────────────────────

USER_AGENT = "TestAgent/1.0"


def make_worker_pool() -> MagicMock:
    wp = MagicMock()
    wp.throttle.return_value.__aenter__ = AsyncMock(return_value=None)
    wp.throttle.return_value.__aexit__ = AsyncMock(return_value=False)
    wp.executor = None  # None → asyncio uses the default thread pool for run_in_executor
    return wp


def make_scraper(urls, backend="bs") -> Scraper:
    return Scraper(urls, USER_AGENT, backend, make_worker_pool())


def make_bs_session(content: bytes = b"", status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.encoding = "utf-8"
    resp.status_code = status_code
    session = MagicMock()
    session.get.return_value = resp
    return session


# ═══════════════════════════════════════════════════════════════════════════
# 1. Scraper.get_scraper() — URL routing
# ═══════════════════════════════════════════════════════════════════════════


class TestGetScraper:
    def setup_method(self):
        self.scraper = make_scraper([])

    def test_pdf_url_routes_to_pymupdf(self):
        from gpt_researcher.scraper.scraper import PyMuPDFScraper as _P
        assert self.scraper.get_scraper("https://example.com/report.pdf") is _P

    def test_arxiv_url_routes_to_arxiv(self):
        from gpt_researcher.scraper.scraper import ArxivScraper as _A
        assert self.scraper.get_scraper("https://arxiv.org/abs/1234.5678") is _A

    def test_regular_url_uses_configured_backend(self):
        from gpt_researcher.scraper.scraper import BeautifulSoupScraper as _BS
        s = make_scraper([], backend="bs")
        assert s.get_scraper("https://example.com/article") is _BS

    def test_pdf_takes_priority_over_backend(self):
        """Even if backend is 'bs', a .pdf link must go to PyMuPDF."""
        from gpt_researcher.scraper.scraper import PyMuPDFScraper as _P
        s = make_scraper([], backend="bs")
        assert s.get_scraper("https://cdn.example.com/data.pdf") is _P

    def test_unknown_backend_raises(self):
        s = make_scraper([], backend="unknown_backend")
        with pytest.raises(Exception, match="Scraper not found"):
            s.get_scraper("https://example.com")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Scraper.extract_data_from_url()
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractDataFromUrl:
    def setup_method(self):
        self.scraper = make_scraper([])

    def _sync_scraper(self, return_value=None, side_effect=None) -> MagicMock:
        """
        A mock scraper instance that only exposes `scrape` (no scrape_async).
        MagicMock auto-creates scrape_async otherwise, making hasattr() return True
        and causing 'MagicMock can't be used in await expression'.
        """
        inst = MagicMock(spec=["scrape"])
        if side_effect is not None:
            inst.scrape.side_effect = side_effect
        else:
            inst.scrape.return_value = return_value
        return inst

    def _patch_scraper_class(self, instance: MagicMock):
        """Return a factory that always yields the given mock instance."""
        return lambda url, session: instance

    @pytest.mark.asyncio
    async def test_short_content_returns_raw_content_none(self):
        """< 100 chars → the 'Content too short or empty' path → raw_content: None."""
        mock_inst = self._sync_scraper(return_value=("Too short.", [], "Title"))

        with patch.object(self.scraper, "get_scraper", return_value=self._patch_scraper_class(mock_inst)):
            result = await self.scraper.extract_data_from_url(
                "https://short.example.com", MagicMock()
            )

        assert result["raw_content"] is None
        assert result["url"] == "https://short.example.com"
        assert result["image_urls"] == []

    @pytest.mark.asyncio
    async def test_99_chars_is_too_short(self):
        """Threshold is strict < 100; 99 chars → raw_content None."""
        mock_inst = self._sync_scraper(return_value=("x" * 99, [], "T"))

        with patch.object(self.scraper, "get_scraper", return_value=self._patch_scraper_class(mock_inst)):
            result = await self.scraper.extract_data_from_url(
                "https://edge.example.com", MagicMock()
            )

        assert result["raw_content"] is None

    @pytest.mark.asyncio
    async def test_101_chars_passes_threshold(self):
        """101 chars (> 100) → raw_content is populated."""
        mock_inst = self._sync_scraper(return_value=("x" * 101, [], "T"))

        with patch.object(self.scraper, "get_scraper", return_value=self._patch_scraper_class(mock_inst)):
            result = await self.scraper.extract_data_from_url(
                "https://edge101.example.com", MagicMock()
            )

        assert result["raw_content"] == "x" * 101

    @pytest.mark.asyncio
    async def test_read_timeout_returns_raw_content_none(self):
        """ReadTimeout (logged as 'Error! : HTTPSConnectionPool... Read timed out') → None."""
        mock_inst = self._sync_scraper(side_effect=requests.exceptions.ReadTimeout("timed out"))

        with patch.object(self.scraper, "get_scraper", return_value=self._patch_scraper_class(mock_inst)):
            result = await self.scraper.extract_data_from_url(
                "https://timeout.example.com", MagicMock()
            )

        assert result["raw_content"] is None
        assert result["url"] == "https://timeout.example.com"

    @pytest.mark.asyncio
    async def test_ssl_error_returns_raw_content_none(self):
        """SSLEOFError (seen in logs for ibm.com) → raw_content: None, no crash."""
        mock_inst = self._sync_scraper(side_effect=requests.exceptions.SSLError("SSL EOF"))

        with patch.object(self.scraper, "get_scraper", return_value=self._patch_scraper_class(mock_inst)):
            result = await self.scraper.extract_data_from_url(
                "https://ssl-error.example.com", MagicMock()
            )

        assert result["raw_content"] is None

    @pytest.mark.asyncio
    async def test_good_content_returned_intact(self):
        """Content > 100 chars → raw_content, image_urls, title all populated."""
        good_content = "A" * 201
        mock_inst = self._sync_scraper(
            return_value=(good_content, [{"url": "img.png", "score": 1}], "Good Title")
        )

        with patch.object(self.scraper, "get_scraper", return_value=self._patch_scraper_class(mock_inst)):
            result = await self.scraper.extract_data_from_url(
                "https://good.example.com", MagicMock()
            )

        assert result["raw_content"] == good_content
        assert result["title"] == "Good Title"
        assert result["image_urls"] == [{"url": "img.png", "score": 1}]

    @pytest.mark.asyncio
    async def test_async_scraper_awaited_not_run_in_executor(self):
        """Scrapers that expose scrape_async() must be awaited, not thread-pooled."""
        good_content = "B" * 150
        # This mock explicitly has scrape_async so hasattr() returns True
        mock_inst = MagicMock(spec=["scrape_async"])
        mock_inst.scrape_async = AsyncMock(return_value=(good_content, [], "Async Title"))

        with patch.object(self.scraper, "get_scraper", return_value=self._patch_scraper_class(mock_inst)):
            result = await self.scraper.extract_data_from_url(
                "https://async.example.com", MagicMock()
            )

        assert result["raw_content"] == good_content
        mock_inst.scrape_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generic_exception_returns_raw_content_none(self):
        """Any unexpected exception must not propagate — return None gracefully."""
        mock_inst = self._sync_scraper(side_effect=RuntimeError("unexpected failure"))

        with patch.object(self.scraper, "get_scraper", return_value=self._patch_scraper_class(mock_inst)):
            result = await self.scraper.extract_data_from_url(
                "https://crash.example.com", MagicMock()
            )

        assert result["raw_content"] is None
        assert result["url"] == "https://crash.example.com"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Scraper.run() — None filtering
# ═══════════════════════════════════════════════════════════════════════════


class TestScraperRun:
    @pytest.mark.asyncio
    async def test_none_results_filtered_out(self):
        """run() must drop every result where raw_content is None."""
        urls = ["https://good.com", "https://empty.com", "https://also-good.com"]
        scraper = make_scraper(urls)

        async def fake_extract(url, session):
            if "empty" in url:
                return {"url": url, "raw_content": None, "image_urls": [], "title": ""}
            return {"url": url, "raw_content": "Content " * 20, "image_urls": [], "title": "T"}

        scraper.extract_data_from_url = fake_extract
        results = await scraper.run()

        assert len(results) == 2
        assert all(r["raw_content"] is not None for r in results)
        assert not any("empty" in r["url"] for r in results)

    @pytest.mark.asyncio
    async def test_all_failed_returns_empty_list(self):
        """If every URL yields raw_content None, run() returns []."""
        urls = ["https://a.com", "https://b.com"]
        scraper = make_scraper(urls)

        async def fake_extract(url, session):
            return {"url": url, "raw_content": None, "image_urls": [], "title": ""}

        scraper.extract_data_from_url = fake_extract
        assert await scraper.run() == []

    @pytest.mark.asyncio
    async def test_empty_url_list_returns_empty(self):
        scraper = make_scraper([])
        assert await scraper.run() == []

    @pytest.mark.asyncio
    async def test_all_succeeded_all_returned(self):
        urls = ["https://x.com", "https://y.com"]
        scraper = make_scraper(urls)

        async def fake_extract(url, session):
            return {"url": url, "raw_content": "Good " * 30, "image_urls": [], "title": "T"}

        scraper.extract_data_from_url = fake_extract
        results = await scraper.run()
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════════════════
# 4. BeautifulSoupScraper — failure modes seen in logs
# ═══════════════════════════════════════════════════════════════════════════


class TestBeautifulSoupScraper:
    def test_read_timeout_returns_empty_triple(self):
        """'Error! : HTTPSConnectionPool... Read timed out' → ("", [], "")."""
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ReadTimeout("timed out")
        bs = BeautifulSoupScraper("https://timeout.example.com", session)

        content, images, title = bs.scrape()
        assert content == ""
        assert images == []
        assert title == ""

    def test_ssl_eof_returns_empty_triple(self):
        """'SSLEOFError... EOF occurred in violation of protocol' → ("", [], "")."""
        session = MagicMock()
        session.get.side_effect = requests.exceptions.SSLError("SSL EOF")
        bs = BeautifulSoupScraper("https://ssl.example.com", session)

        content, images, title = bs.scrape()
        assert content == ""
        assert images == []
        assert title == ""

    def test_connection_error_returns_empty_triple(self):
        session = MagicMock()
        session.get.side_effect = requests.exceptions.ConnectionError("refused")
        bs = BeautifulSoupScraper("https://refused.example.com", session)

        content, images, title = bs.scrape()
        assert content == ""
        assert images == []
        assert title == ""

    def test_good_html_returns_content_and_title(self):
        html = (
            b"<html><head><title>AI Jobs Report</title></head>"
            b"<body><p>Artificial intelligence is reshaping the employment market "
            b"significantly in 2026 across many industries and sectors worldwide.</p></body></html>"
        )
        session = make_bs_session(html)
        bs = BeautifulSoupScraper("https://good.example.com", session)
        content, images, title = bs.scrape()

        assert len(content) > 0
        assert title == "AI Jobs Report"

    def test_script_and_style_stripped_from_content(self):
        html = (
            b"<html><head><title>T</title>"
            b"<style>body{color:red;font-size:12px}</style></head>"
            b"<body><script>alert('xss')</script>"
            b"<p>Real article content about the employment market and AI.</p></body></html>"
        )
        session = make_bs_session(html)
        bs = BeautifulSoupScraper("https://strip.example.com", session)
        content, _, _ = bs.scrape()

        assert "alert" not in content
        assert "color:red" not in content
        assert "Real article content" in content

    def test_empty_body_returns_minimal_content(self):
        """Completely empty body → content is empty or whitespace-only."""
        session = make_bs_session(b"<html><body></body></html>")
        bs = BeautifulSoupScraper("https://empty.example.com", session)
        content, images, title = bs.scrape()
        assert content.strip() == ""


# ═══════════════════════════════════════════════════════════════════════════
# 5. PyMuPDFScraper — is_url + failure modes
# ═══════════════════════════════════════════════════════════════════════════


class TestPyMuPDFScraper:

    # ── is_url ───────────────────────────────────────────────────────────

    def test_is_url_valid_https(self):
        assert PyMuPDFScraper("https://example.com/doc.pdf").is_url() is True

    def test_is_url_valid_http(self):
        assert PyMuPDFScraper("http://example.com/doc.pdf").is_url() is True

    def test_is_url_local_path(self):
        assert PyMuPDFScraper("/tmp/local.pdf").is_url() is False

    def test_is_url_empty_string(self):
        assert PyMuPDFScraper("").is_url() is False

    # ── Failure modes ────────────────────────────────────────────────────

    def test_download_timeout_returns_empty(self):
        """PDF download timeout (seen for pdf.dfcfw.com URLs in logs) → ("", [], "")."""
        s = PyMuPDFScraper("https://pdf.example.com/report.pdf")
        with patch(
            "gpt_researcher.scraper.pymupdf.pymupdf.requests.get",
            side_effect=requests.exceptions.Timeout("timed out"),
        ):
            content, images, title = s.scrape()
        assert content == ""
        assert images == []
        assert title == ""

    def test_http_404_returns_empty(self):
        """Non-2xx response → ("", [], "")."""
        s = PyMuPDFScraper("https://pdf.example.com/missing.pdf")
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("404")
        with patch("gpt_researcher.scraper.pymupdf.pymupdf.requests.get", return_value=mock_resp):
            content, images, title = s.scrape()
        assert content == ""
        assert title == ""

    def test_empty_doc_returns_empty(self):
        """PyMuPDFLoader returning [] (corrupt / unreadable PDF) → ("", [], "")."""
        s = PyMuPDFScraper("https://pdf.example.com/corrupt.pdf")

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = iter([b"fake bytes"])

        mock_file = MagicMock()
        mock_file.name = "/tmp/fake.pdf"
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_file)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("gpt_researcher.scraper.pymupdf.pymupdf.requests.get", return_value=mock_resp),
            patch("gpt_researcher.scraper.pymupdf.pymupdf.tempfile.NamedTemporaryFile", return_value=mock_ctx),
            patch("gpt_researcher.scraper.pymupdf.pymupdf.PyMuPDFLoader") as MockLoader,
            patch("gpt_researcher.scraper.pymupdf.pymupdf.os.remove"),
        ):
            MockLoader.return_value.load.return_value = []
            content, images, title = s.scrape()

        assert content == ""
        assert title == ""

    def test_successful_pdf_returns_content(self):
        """Successful PDF scrape → content from all pages, title from first page metadata."""
        s = PyMuPDFScraper("https://pdf.example.com/good.pdf")

        mock_page = MagicMock()
        mock_page.page_content = "AI impact on employment and wage polarization study 2026."
        mock_page.metadata = {"title": "AI Employment Report"}

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = iter([b"pdf bytes"])

        mock_file = MagicMock()
        mock_file.name = "/tmp/fake.pdf"
        mock_ctx = MagicMock()
        mock_ctx.__enter__ = MagicMock(return_value=mock_file)
        mock_ctx.__exit__ = MagicMock(return_value=False)

        with (
            patch("gpt_researcher.scraper.pymupdf.pymupdf.requests.get", return_value=mock_resp),
            patch("gpt_researcher.scraper.pymupdf.pymupdf.tempfile.NamedTemporaryFile", return_value=mock_ctx),
            patch("gpt_researcher.scraper.pymupdf.pymupdf.PyMuPDFLoader") as MockLoader,
            patch("gpt_researcher.scraper.pymupdf.pymupdf.os.remove"),
        ):
            MockLoader.return_value.load.return_value = [mock_page]
            content, images, title = s.scrape()

        assert "AI impact" in content
        assert title == "AI Employment Report"
        assert images == []

    def test_local_pdf_load(self):
        """Local file path skips HTTP download and goes straight to PyMuPDFLoader."""
        s = PyMuPDFScraper("/tmp/local.pdf")

        mock_page = MagicMock()
        mock_page.page_content = "Local PDF content about labor market analysis."
        mock_page.metadata = {"title": "Local Report"}

        with patch("gpt_researcher.scraper.pymupdf.pymupdf.PyMuPDFLoader") as MockLoader:
            MockLoader.return_value.load.return_value = [mock_page]
            content, images, title = s.scrape()

        assert "Local PDF content" in content
        assert title == "Local Report"
