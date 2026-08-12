#!/usr/bin/env bash
# The retrial demo, as a runnable script.
#
# This is the source of truth for the GIF: record yourself running it, or let
# VHS drive it (see demo/README.md). It uses its own scratch database so it
# never touches the store in your working tree, and it starts from empty every
# time so the output is the same on every take.
#
#   ./demo/demo.sh
#
# Requires the scripted example only - no API key, no spend.

set -euo pipefail

RETRIAL="${RETRIAL:-retrial}"
export RETRIAL_DB="${RETRIAL_DB:-$(pwd)/demo/demo.db}"

rm -f "$RETRIAL_DB"

pause() { sleep "${PAUSE:-1.5}"; }

# 1. Record a run. The decorator logs every step as the loop runs; there is no
#    export step.
python examples/booking_agent.py
pause

# 2. Every step has a content-hash SHA, addressable by a short prefix like git.
SESSION=$($RETRIAL list | grep -o 's_[a-f0-9]*' | head -1)
$RETRIAL log "$SESSION"
pause

# 3. The flight cost $450. What would it have done at $1200?
#    `2b04ebc`-style SHAs change per run, so resolve the search_flight step.
STEP=$($RETRIAL log "$SESSION" | grep -B1 'ran search_flight' | grep -o '^  [a-f0-9]\{7\}' | tr -d ' ')
cat examples/edit_price.json
pause

# 4. Fork: replay up to that step, splice in $1200, re-enter the real loop.
FORK=$($RETRIAL fork "$STEP" \
        --agent examples.booking_agent:run_agent \
        --edit-file examples/edit_price.json \
      | grep -o 's_[a-f0-9]*' | head -1)
echo "Forked into session $FORK"
pause

# 5. Where did the two runs diverge, and what did each finally say?
$RETRIAL diff "$SESSION" "$FORK"
pause

# 6. What did it cost? The scripted model is unknown to the price table, so
#    retrial reports `unpriced` rather than guessing.
$RETRIAL cost "$SESSION"
