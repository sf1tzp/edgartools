"""Regression: Cross-Reference Index item lookups return text, not raw HTML.

Cluster-4 sweep (CITIGROUP INC 10-K, accession 0000831001-23-000037). Citigroup
organizes its 10-K by descriptive titles and maps SEC items only through a "FORM
10-K CROSS-REFERENCE INDEX" table (item -> page ranges). Section detection finds
nothing (no "Item N" headers, no per-item anchors), so `obj['Item 7A']` resolves
through the cross-reference index.

The bug: the index's page-range extraction returned raw *source HTML*
(`self.html[start:end]`), and Item 7A spans ~200 heavily inline-styled pages —
so `obj['Item 7A']` returned **12,070,332 chars of HTML**, 14x the filing's own
extracted full text (859,373 chars), enough to blow an LLM context window. Every
other item lookup on the object returns text.

The fix: `CrossReferenceIndex.extract_item_text` renders each page-range slice to
plain text, and `TenK.__getitem__` calls it instead of `extract_item_content`
(which keeps its raw-HTML contract for callers that want markup). Item 7A now
returns ~540K chars of text, under the filing's full-text length.
"""
import pytest

from edgar.documents import CrossReferenceIndex

# Minimal cross-reference-index filing: the heading, a one-row index table
# (Item 1A -> page 2), and inline-styled body content on page 2.
_SYNTHETIC_HTML = """
<html><body>
<div><span style="font-weight:700">FORM 10-K CROSS-REFERENCE INDEX</span></div>
<table>
  <tr><td>1A.</td><td>Risk Factors</td><td>2</td></tr>
</table>
<hr style="page-break-after:always"/>
<div><span style="font-weight:700;color:#111">RISK FACTORS.</span>
     <span style="color:#000000;font-family:'Times New Roman'">Our business faces risks.</span></div>
<hr style="page-break-after:always"/>
<div>Unrelated page 3 content.</div>
</body></html>
"""


def test_extract_item_text_strips_markup():
    """extract_item_text returns clean text; extract_item_content keeps HTML."""
    idx = CrossReferenceIndex(_SYNTHETIC_HTML)
    assert idx.has_index()

    text = idx.extract_item_text('1A')
    assert text is not None
    # Text, not markup.
    assert '<' not in text and '>' not in text
    assert 'RISK FACTORS.' in text
    assert 'Our business faces risks.' in text
    # Bounded to the mapped page — the unrelated page 3 must not leak in.
    assert 'Unrelated page 3' not in text

    # The HTML method still returns markup for callers that want it.
    html = idx.extract_item_content('1A')
    assert html is not None and '<span' in html


def test_extract_item_text_missing_item_is_none():
    """An item absent from the index returns None, not an empty-string slice."""
    idx = CrossReferenceIndex(_SYNTHETIC_HTML)
    assert idx.extract_item_text('7A') is None


@pytest.mark.network
def test_citigroup_item7a_is_text_not_raw_html():
    """CITIGROUP Item 7A returns bounded text, not a 12M-char HTML blob."""
    from edgar import find

    obj = find("0000831001-23-000037").obj()
    full_text_len = len(obj._filing.text())

    item7a = obj['Item 7A']
    assert item7a is not None
    # The defect: 12,070,332 chars of raw HTML, 14x the filing's own text.
    assert '</div>' not in item7a[-500:] and '<span' not in item7a[:5000], \
        "Item 7A is returning raw HTML"
    assert len(item7a) < full_text_len, (
        f"Item 7A ({len(item7a)} chars) exceeds the filing's full text "
        f"({full_text_len}) — raw-HTML over-capture")
    # Still substantial market-risk content, not empty.
    assert len(item7a) > 50_000
