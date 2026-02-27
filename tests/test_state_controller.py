import json

from multi_agents.agents.state_controller import StateController


def test_state_controller_atomic_update(tmp_path):
    state_file = tmp_path / "agent_state.json"
    controller = StateController(str(state_file))

    assert state_file.exists()
    assert controller.get_current_idx("scrap") == 3

    controller.set_tier("scrap", 2)
    assert controller.get_current_idx("scrap") == 2

    controller.promote("scrap")
    assert controller.get_current_idx("scrap") == 1

    controller.demote("scrap")
    assert controller.get_current_idx("scrap") == 2

    reloaded = StateController(str(state_file))
    assert reloaded.get_current_idx("scrap") == 2
    assert isinstance(reloaded.get_current_model("scrap"), str)

    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert data["scrap"]["current_idx"] == 2
