# Reading IM77 sign sequences from indusscript.in (Browse page)

You read the sign sequence of each IM77 text number in your chunk file by using the site's normal
Browse page in the user's Chrome (Claude in Chrome tools, mcp__claude-in-chrome__*). The user is
signed in already and asked for this. Use the site only as a person would: type a number, look,
hover over signs. Never run JavaScript on the page, never read its network traffic, never query
its database or any API. One lookup at a time, no parallel tabs.

Folder: /tmp/claude-0/-home-claude-indus-computational-epigraphy/f0a2f271-e802-58a3-8d3c-789975153be6/scratchpad/cisi
Helper: `python lookup/../glyph_points.py SCREENSHOT` (run from that folder as `python glyph_points.py PATH`)

## Setup (once)
1. Load tools in one ToolSearch call: select:mcp__claude-in-chrome__tabs_context_mcp,mcp__claude-in-chrome__computer,mcp__claude-in-chrome__browser_batch
2. Call tabs_context_mcp. Use the tab whose URL is https://indusscript.in/ (do not create new tabs).
   The banner and the nav bar at the top are FIXED and cover y 0 to ~345; never click there
   (banner images are links to other sections).

## For each number N in your chunk (in order)
A. browser_batch:
   - scroll up 10 ticks at (768,400)          [page top; CODES boxes now at y≈434]
   - left_click (894,434)                      [last CODES box]
   - key "BackSpace" repeat 6                  [clears all four boxes]
   - type N  (4 digits, keep leading zeros)
   - wait 3
   - zoom region (600,400,1000,530) scale 0.4  [badge: "N found" or "N does not exist"]
   - scroll down 3 ticks at (768,500)
   - wait 3                                     [the card shows a spinner first]
   - screenshot with save_to_disk true, scale 0.6
   Note the saved path printed in the result. (Scroll actions also return screenshots; ignore them.)
B. If the badge said "does not exist", record NOTFOUND and go on (the card below then shows some
   other text: never record it). Otherwise check the centred card shows N in its top-left badge;
   if the card is still loading, wait 2 and take the screenshot again.
C. Bash: `python glyph_points.py <saved path>` → JSON with `points` (hover points, left to right),
   `runs` (raw ink pieces), `footer_zoom`, `glyph_zoom`. If card is null, wait 2, repeat A once,
   then record ERROR if it still fails.
D. browser_batch:
   - zoom footer_zoom region, scale 0.8 → read "Length:L" (number of signs) and nothing else needed
   - for each point [x,y] in points: hover (x,y); wait 1; zoom glyph_zoom region, scale 0.7
     → read the red tooltip "Sign: K" under the highlighted sign.
E. Check: you need exactly L sign numbers. If the points gave fewer than L (two signs sitting close
   together were one point), hover at the centre of each raw run inside that cluster ((x0+x1)/2, same y)
   and read the tooltips; consecutive runs that show the same tooltip on the same highlighted box are
   ONE sign. Two adjacent identical signs have two different highlight boxes. If you still cannot get
   exactly L, record what you have and put CHECK in notes.
F. Note from the card header: "Side: s Line: l". If the card partly visible on the right edge carries
   the same number N, write MULTILINE in notes (we only read the centred card).
G. Append one line to your results file with Bash (`echo ... >> FILE`):
   `N,L,K1 K2 ... KL,side/line,notes`
   where K1..KL are the sign numbers in the order they appear on the card from LEFT to RIGHT.
   Example for N=2177: `2177,5,1 342 48 99 267,0/0,`

Do not guess a sign number you did not see in a tooltip. If a tooltip was unreadable, hover again.
If the browser stops responding or tools error 3 times in a row, stop and report where you stopped.
