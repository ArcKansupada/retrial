# Recording the demo

`demo.sh` is the source of truth for what the demo shows. Everything here is
just a way of filming it. It uses its own scratch database (`demo/demo.db`) and
wipes it on every run, so takes are repeatable and your real store is untouched.

```bash
./demo/demo.sh            # watch it once before recording
```

The sequence: record a run -> `log` (SHAs visible) -> the edit -> `fork` ->
`diff`, ending on the two different final answers. About 25 seconds at a
readable pace.

## Option 1 - screen recorder (most reliable on Windows)

[ScreenToGif](https://www.screentogif.com/) is free, native, and has a built-in
editor for trimming and setting frame delays.

1. Open a terminal, size it to about 100x30, and bump the font to ~16pt.
2. Start recording the terminal window only.
3. Run the commands from `demo.sh` by hand, pausing on `log` and `diff` so a
   viewer can actually read them.
4. Trim the dead air, export at 12-15 fps.

Typing by hand looks better than a script here - the small hesitations read as
real, and you can linger on the interesting output.

## Option 2 - VHS (scripted, reproducible)

[VHS](https://github.com/charmbracelet/vhs) renders a GIF from `demo.tape`, so
re-recording after a change is one command:

```bash
vhs demo/demo.tape        # -> demo/demo.gif
```

Works cleanly on macOS, Linux, and WSL. On native Windows the bash shell
handling is fragile - if it misbehaves, run it under WSL or fall back to
Option 1.

The tape captures two ids into shell variables, because session ids are
generated per run and a tape cannot know the fork's id in advance. If you want
the on-screen commands to be free of `$(...)`, record by hand instead.

## Option 3 - asciinema (text, not a GIF)

```bash
asciinema rec demo.cast
```

Selectable text and tiny files, but it needs a JS player, so it will not embed
in every blog host and dies in RSS. Good as a companion to a GIF, not a
replacement.

## Making it readable

- **Font size over resolution.** A 1280px GIF at 15pt beats 1920px at 11pt;
  most people watch it inline at half size.
- **Pause on `log` and `diff`.** They are the two frames carrying the argument.
  Three to four seconds each, longer than feels natural while recording.
- **End on the final answers.** `Confirmed: ... booked for $450` above
  `That's over the $600 limit` is the whole product in two lines. Hold it.
- **Caption it**, so it stands alone for anyone who will not press play:
  *forking a $450 fare with $1,200 spliced in - the agent declines to book, and
  calls a tool the original run never reached.*
