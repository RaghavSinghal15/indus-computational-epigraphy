"""Run the card reader over capture logs and write lookup/auto_results.csv."""
import csv, sys
from read_card2 import read
rows = []
latest = {}                                   # keep the last capture per number (later runs win)
for log in sys.argv[1:]:
    for line in open(log):
        parts = line.strip().split(',')
        if len(parts) >= 3:
            latest[parts[0]] = parts
for N, parts in latest.items():
        status, path = parts[1], parts[2]
        if status != 'found' or not path:
            rows.append(dict(im77=N, length='', signs_left_to_right='', min_margin='', notes='NOTFOUND')); continue
        r = read(path)
        if not r['ok']:
            rows.append(dict(im77=N, length='', signs_left_to_right='', min_margin='', notes='READFAIL ' + r['why'])); continue
        mm = min(r['margins'])
        rows.append(dict(im77=N, length=r['length'], signs_left_to_right=' '.join(map(str, r['signs_left_to_right'])),
                         min_margin=round(mm, 3), notes=('LENGUESS ' if r.get('length_guessed') else '') + ('LOWMARGIN' if mm < 0.05 else '')))
with open('lookup/auto_results.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows), Counter(r['notes'].split(' ')[0] for r in rows))
