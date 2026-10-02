"""Guard: user-facing text must survive a cp1252 console.

The one exception is `_glyphs()`, which checks encodability itself.
"""

import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).parent.parent / "retrial"
SANCTIONED = {"cli.py": {"_glyphs"}}


def source_files():
    return sorted(PACKAGE.glob("*.py"))


@pytest.mark.parametrize("path", source_files(), ids=lambda p: p.name)
def test_source_is_ascii_except_the_guarded_glyphs(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    inside_glyphs = False
    offenders = []

    for number, line in enumerate(lines, 1):
        if line.startswith("def "):
            inside_glyphs = any(
                line.startswith(f"def {name}(")
                for name in SANCTIONED.get(path.name, ())
            )
        if inside_glyphs:
            continue
        if any(ord(char) > 127 for char in line):
            offenders.append(number)

    assert not offenders, (
        f"{path.name} has non-ASCII on line(s) {offenders}. These garble on a "
        "cp1252 console. Use '-' instead of an em-dash, and plain quotes."
    )


def test_every_string_the_cli_can_print_encodes_as_cp1252():
    """Belt and braces: the whole module must round-trip through cp1252."""
    source = (PACKAGE / "cli.py").read_text(encoding="utf-8")
    non_ascii = {c for c in source if ord(c) > 127}
    # Only the box-drawing set, and only because _glyphs() gates it.
    assert non_ascii <= set("└─├│")


def test_glyphs_are_actually_gated():
    """If the guard is ever removed, the exception above must stop applying."""
    source = (PACKAGE / "cli.py").read_text(encoding="utf-8")
    assert "except (UnicodeEncodeError, LookupError):" in source


# --- model output is not ours to police ------------------------------------
#
# Text the model wrote can be any Unicode, and retrial prints it.


class FakeConsole:
    """A writable stream reporting a specific terminal encoding.

    Not io.StringIO, whose `encoding` is read-only.
    """

    def __init__(self, encoding):
        self.encoding = encoding
        self._chunks = []

    def write(self, text):
        self._chunks.append(text)
        return len(text)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self._chunks)


def _echo_to(monkeypatch, encoding, text):
    from retrial.cli import echo

    console = FakeConsole(encoding)
    monkeypatch.setattr("retrial.cli.sys.stdout", console)
    monkeypatch.setattr("click.utils._default_text_stdout", lambda: console)
    echo(text)
    return console.getvalue()


def test_echo_survives_model_output_a_cp1252_console_cannot_encode(monkeypatch):
    # Check mark, euro sign, airplane - the shape of a real Opus booking reply.
    out = _echo_to(monkeypatch, "cp1252", "Booked ✅ for €450 \U0001f6eb")
    assert "Booked" in out and "450" in out  # the message survived
    out.encode("cp1252")  # and it is printable on this console


def test_echo_leaves_encodable_text_untouched(monkeypatch):
    out = _echo_to(monkeypatch, "cp1252", "Confirmed: AUS-SFO booked for $450.")
    assert out == "Confirmed: AUS-SFO booked for $450.\n"


def test_echo_passes_unicode_through_on_a_utf8_console(monkeypatch):
    """A capable terminal still gets the real characters."""
    out = _echo_to(monkeypatch, "utf-8", "Booked ✅")
    assert "✅" in out


#: The only two calls allowed to reach click directly: the bodies of echo() and warn().
GUARDED_CALLS = ("click.echo(text)", "click.echo(text, err=True)")


def test_every_cli_print_goes_through_echo():
    """One raw click.echo would reopen the hole."""
    source = (PACKAGE / "cli.py").read_text(encoding="utf-8")
    raw = [
        number
        for number, line in enumerate(source.splitlines(), 1)
        if "click.echo(" in line and not any(call in line for call in GUARDED_CALLS)
    ]
    assert not raw, (
        f"cli.py calls click.echo directly on line(s) {raw}. Use echo() or warn() "
        "so model output can't crash the command printing it."
    )


def test_warn_degrades_like_echo(monkeypatch):
    """warn() is echo() for stderr, and gets the same protection."""
    from retrial.cli import warn

    console = FakeConsole("cp1252")
    monkeypatch.setattr("retrial.cli.sys.stderr", console)
    monkeypatch.setattr("click.utils._default_text_stderr", lambda: console)

    warn("Booked ✅ for €450 \U0001f6eb")

    out = console.getvalue()
    assert "Booked" in out and "450" in out
    out.encode("cp1252")


def test_warn_leaves_encodable_text_untouched(monkeypatch):
    from retrial.cli import warn

    console = FakeConsole("cp1252")
    monkeypatch.setattr("retrial.cli.sys.stderr", console)
    monkeypatch.setattr("click.utils._default_text_stderr", lambda: console)

    warn("note: a store already exists above this directory")

    assert console.getvalue() == "note: a store already exists above this directory\n"


def test_export_data_deliberately_bypasses_echo():
    """The one place that must NOT degrade.

    An export is data: a replaced character would break the step's sha.
    """
    import ast

    source = (PACKAGE / "cli.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    write_data = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "write_data"
    )

    # The AST, not the text, so the docstring's own wording cannot trip the check.
    called = {
        node.func.id
        for node in ast.walk(write_data)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "echo" not in called and "warn" not in called

    utf8 = [
        node.value
        for node in ast.walk(write_data)
        if isinstance(node, ast.Constant) and node.value == "utf-8"
    ]
    assert len(utf8) >= 2, "both the file and the stdout path must name utf-8"
