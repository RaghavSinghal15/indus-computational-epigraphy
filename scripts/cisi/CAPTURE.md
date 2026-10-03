# Capture card screenshots from indusscript.in (Browse page)

Use the site only as a person would (type a number, scroll, take screenshots) in the user's Chrome
via Claude in Chrome tools. The user is signed in and asked for this. Never run JavaScript in the
page, never inspect network requests, never query any database or API. One tab, one lookup at a time.

Setup: ToolSearch "select:mcp__claude-in-chrome__tabs_context_mcp,mcp__claude-in-chrome__browser_batch",
then tabs_context_mcp; use the tab with URL https://indusscript.in/. If the page is not on the BROWSE
tab, scroll up 10 at (768,400) and click "BROWSE" in the menu bar (left side, y≈324).

For each number N in your list, ONE browser_batch:
  1. scroll up 10 ticks at (768,400)
  2. left_click (894,434)            (last CODES box; the boxes sit at y≈434 when the page is at the top)
  3. key "BackSpace" repeat 6
  4. type N
  5. wait 3
  6. zoom region (600,400,1000,530) scale 0.4     -> badge "N found" / "N does not exist"
  7. scroll down 3 ticks at (768,500)
  8. wait 3
  9. screenshot save_to_disk true (scale 1)
Then append ONE line to your output file with Bash:  N,<badge: found|missing>,<saved screenshot path>
(the path printed for step 9; the scroll steps also print paths, ignore those).
If the badge said "does not exist" write missing and an empty path.
If the step-9 screenshot shows the card still loading (spinner, no signs), take one more
screenshot (wait 2 first) and use that path instead.
Do not read the signs yourself; a script does that later.
If a call fails ("Couldn't determine which page", timeout, or "extension disconnected"), wait about 10 s (computer wait action or a Bash sleep 10), call tabs_context_mcp
and retry; the extension usually reconnects by itself. Only after 6 failures in a row stop and report the number you stopped at. Always redo an interrupted number from step 1.
