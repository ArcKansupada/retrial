import pytest

from retrial.patch import PatchError, apply_patch, normalize_edit

STEP = {
    "sha": "abc",
    "type": "tool_call",
    "output": [{"tool_use_id": "toolu_01", "content": '{"flight_price": 450}'}],
}


def test_replace_a_nested_field():
    out = apply_patch(STEP, {"op": "replace", "path": "/output/0/content", "value": "x"})
    assert out["output"][0]["content"] == "x"


def test_patch_never_mutates_the_input():
    apply_patch(STEP, {"op": "replace", "path": "/output/0/content", "value": "x"})
    assert STEP["output"][0]["content"] == '{"flight_price": 450}'


def test_multiple_ops_apply_in_order():
    out = apply_patch(
        STEP,
        [
            {"op": "replace", "path": "/output/0/content", "value": "a"},
            {"op": "add", "path": "/output/0/note", "value": "edited"},
        ],
    )
    assert out["output"][0] == {
        "tool_use_id": "toolu_01",
        "content": "a",
        "note": "edited",
    }


def test_add_appends_to_a_list():
    out = apply_patch(STEP, {"op": "add", "path": "/output/-", "value": {"x": 1}})
    assert out["output"][-1] == {"x": 1}


def test_remove():
    out = apply_patch(STEP, {"op": "remove", "path": "/output/0"})
    assert out["output"] == []


def test_json_pointer_escapes():
    doc = {"a/b": {"c~d": 1}}
    assert apply_patch(doc, {"op": "replace", "path": "/a~1b/c~0d", "value": 2}) == {
        "a/b": {"c~d": 2}
    }


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"op": "replace", "path": "output/0", "value": 1}, "JSON Pointer"),
        ({"op": "frobnicate", "path": "/output", "value": 1}, "unsupported op"),
        ({"op": "replace", "path": "/nope", "value": 1}, "no existing key"),
        ({"op": "replace", "path": "/output/9/content", "value": 1}, "out of range"),
        ({"op": "replace", "path": "/output/x", "value": 1}, "not a valid list index"),
        ({"op": "replace", "path": "/output"}, "needs a 'value'"),
        ({"op": "replace", "path": "/"}, "cannot patch the step root"),
    ],
)
def test_bad_patches_fail_loudly(patch, message):
    """A typo'd path must raise, not silently no-op."""
    with pytest.raises(PatchError, match=message):
        apply_patch(STEP, patch)


# --- normalize_edit: the hybrid API ----------------------------------------


def test_patch_edits_round_trip_as_provenance():
    patch = {"op": "replace", "path": "/output/0/content", "value": "x"}
    apply, provenance = normalize_edit(patch)
    assert provenance == {"type": "patch", "patch": patch}
    assert apply(STEP)["output"][0]["content"] == "x"


def test_callback_edits_are_recorded_as_not_round_tripping():
    def double_it(step):
        return step

    _, provenance = normalize_edit(double_it)
    assert provenance["type"] == "callback"
    assert provenance["repr"] == "double_it"
    assert "not round-trip" in provenance["note"]


def test_no_edit_is_a_pure_replay():
    apply, provenance = normalize_edit(None)
    assert provenance is None
    assert apply(STEP) == STEP


def test_a_nonsense_edit_is_rejected():
    with pytest.raises(PatchError, match="got int"):
        normalize_edit(42)
