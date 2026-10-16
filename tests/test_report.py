import json

import pytest
from pypdf import PdfReader

from eeg_explorer import pipeline as pl
from eeg_explorer import tools
from eeg_explorer.data import make_demo_recording
from eeg_explorer.pipeline import AnalysisState
from eeg_explorer.report import build_report, markdown_flowables, _styles


@pytest.fixture(scope="module")
def analysed_state() -> AnalysisState:
    state = AnalysisState()
    make_demo_recording(state, n_trials=12, seed=3)
    pl.preprocess(state)
    pl.make_epochs(state)
    pl.localize(state, method="eLORETA")
    pl.peak_sources(state, 0.25, 0.35)
    pl.band_power(state)
    return state


def test_full_report_has_all_sections(analysed_state, tmp_path):
    findings = "## Result\nPeak at **300 ms** in the *left* sensorimotor cortex.\n\n- point one\n- point two"
    info = build_report(analysed_state, findings=findings, title="Test report", author="Lab X",
                        output_dir=tmp_path)
    assert info["sections"] == ["findings", "recording", "processing", "localisation", "spectrum",
                                "ground_truth", "log"]
    assert info["pages"] >= 3
    reader = PdfReader(info["path"])
    assert len(reader.pages) == info["pages"]
    text = "\n".join(p.extract_text() for p in reader.pages)
    for needle in ("Test report", "Lab X", "Summary of findings", "point one", "Source localisation",
                   "Strongest spatially separated sources", "Spectral analysis", "Validation against simulated",
                   "Analysis log", "eLORETA"):
        assert needle in text
    assert reader.metadata.title == "Test report"
    assert analysed_state.reports[-1]["path"] == info["path"]


def test_report_before_localisation_skips_optional_sections(tmp_path):
    state = AnalysisState()
    make_demo_recording(state, n_trials=5)
    info = build_report(state, findings="", output_dir=tmp_path)
    assert info["sections"] == ["findings", "recording"]
    assert info["pages"] == 1


def test_generate_report_tool(analysed_state):
    out, err = tools.run_tool(analysed_state, "generate_report", {"findings": "Short summary."})
    assert not err
    info = json.loads(out)
    assert info["pages"] >= 3 and info["filename"].endswith(".pdf")
    out, err = tools.run_tool(analysed_state, "generate_report", {})
    assert err  # findings is required


def test_markdown_flowables_structure():
    flow = markdown_flowables("# Title\nPara **bold** `code`\n\n- a\n- b\n\n1. c", _styles())
    kinds = [type(f).__name__ for f in flow]
    assert kinds == ["Paragraph", "Paragraph", "ListFlowable", "ListFlowable"]
    assert "<b>bold</b>" in flow[1].text and "Courier" in flow[1].text


def test_report_without_findings_still_builds(analysed_state, tmp_path):
    info = build_report(analysed_state, output_dir=tmp_path)
    text = PdfReader(info["path"]).pages[0].extract_text()
    assert "No analyst narrative" in text
