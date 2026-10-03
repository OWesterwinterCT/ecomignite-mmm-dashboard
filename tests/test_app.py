"""Exercise actual Streamlit widgets, result rendering, and constraint errors."""
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


def test_default_dashboard_and_optimizer():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert "SYNTHETIC DEMO" in app.warning[0].value
    labels = [t.label for t in app.tabs]
    assert {"Data summary & validation", "Model results", "Budget optimizer"}.issubset(labels)
    app.button(key="run_optimizer").click().run(timeout=60)
    assert not app.exception
    assert not app.error
    result = app.session_state["optimizer_result"][1]
    assert len(result["allocation"]) == 8
    assert len(result["weekly_plan"]) == 12
    assert "Estimated total-order uplift" in [m.label for m in app.metric]


def test_configurable_horizon_and_visible_infeasible_budget():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    for widget in app.number_input:
        if widget.label == "Planning weeks":
            widget.set_value(4)
        if widget.label == "Maximum spend change (%)":
            widget.set_value(10.)
    app.run()
    app.button(key="run_optimizer").click().run(timeout=60)
    assert not app.exception and not app.error
    result = app.session_state["optimizer_result"][1]
    assert result["summary"]["horizon_weeks"] == 4
    assert result["summary"]["reference_weeks"] == 4
    for widget in app.checkbox:
        if widget.label == "Use the historical average budget":
            widget.set_value(False)
    app.run()
    for widget in app.number_input:
        if widget.label == "Total budget cap for the whole period":
            widget.set_value(1.)
    app.run()
    app.button(key="run_optimizer").click().run(timeout=60)
    assert not app.exception
    assert app.error
    assert "conflict" in app.error[0].value
