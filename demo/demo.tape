# VHS tape for the retrial demo GIF.  https://github.com/charmbracelet/vhs
#
#   vhs demo/demo.tape        ->  demo/demo.gif
#
# Run from the repo root, with retrial importable (activate your venv first).
# VHS drives a real shell, so what you see is a real run - not a mock-up.
#
# Note on the two `$(...)` captures below: session ids are generated per run,
# so a tape cannot hardcode the fork's id. Capturing it in a variable keeps the
# take honest and repeatable. If you would rather show clean commands, record
# demo.sh by hand instead (see README.md).

Output demo/demo.gif

Set Shell bash
Set FontSize 15
Set Width 1280
Set Height 760
Set Padding 22
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

# Fresh store, so every take is identical. Hidden - it is setup, not content.
Hide
Type "export RETRIAL_DB=$(pwd)/demo/demo.db && rm -f $RETRIAL_DB && clear"
Enter
Sleep 1s
Show

# --- 1. Record a run ------------------------------------------------------
Type "python examples/booking_agent.py"
Enter
Sleep 3s

# --- 2. Every step is addressable by a content-hash SHA -------------------
Type "SESSION=$(retrial list | grep -o 's_[a-f0-9]*' | head -1)"
Enter
Sleep 500ms
Type "retrial log $SESSION"
Enter
Sleep 4s

# --- 3. The counterfactual: what if the fare had been $1200? --------------
Type "cat examples/edit_price.json"
Enter
Sleep 3s

# --- 4. Fork: replay to that step, splice the new fare, re-run for real ---
Type "STEP=$(retrial log $SESSION | grep -B1 'ran search_flight' | grep -oE '^  [a-f0-9]{7}' | tr -d ' ')"
Enter
Sleep 500ms
Type "FORK=$(retrial fork $STEP --agent examples.booking_agent:run_agent --edit-file examples/edit_price.json | grep -o 's_[a-f0-9]*')"
Enter
Sleep 2s
Type "echo forked into $FORK"
Enter
Sleep 2s

# --- 5. The payoff: where they diverged, and what each one decided --------
Type "retrial diff $SESSION $FORK"
Enter
Sleep 8s
