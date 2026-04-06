import json

from multi_agents.agents.state_controller import StateController


def test_state_controller_atomic_update(tmp_path):
    state_file = tmp_path / "agent_state.json"
    controller = StateController(str(state_file))

    assert state_file.exists()
    assert controller.get_current_idx("scraping") == 3

    controller.set_tier("scraping", 2)
    assert controller.get_current_idx("scraping") == 2

    controller.promote("scraping")
    assert controller.get_current_idx("scraping") == 1

    controller.demote("scraping")
    assert controller.get_current_idx("scraping") == 2

    reloaded = StateController(str(state_file))
    assert reloaded.get_current_idx("scraping") == 2
    assert isinstance(reloaded.get_current_model("scraping"), str)

    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["scraping"]["current_idx"] == 2
