"""Run stage 2 for a site, write photos/<site>/*, photos/index.csv (merged), diagnostics."""
import sys, json, csv, os, shutil, glob
import stage2
from stage2 import process_page, crop_and_index, RANGES, VALID

site = sys.argv[1]
pages = [n for n in sorted(RANGES) if RANGES[n][0] == site]
if site in ('X', 'ALL'):
    # remove crops written by an earlier run of these pages
    if os.path.exists('photos/index.csv'):
        for r in csv.DictReader(open('photos/index.csv')):
            if int(r['page']) in set(pages) and os.path.exists('photos/' + r['file']):
                os.remove('photos/' + r['file'])
elif os.path.isdir(f'photos/{site}'):
    shutil.rmtree(f'photos/{site}')
rows = []; diag = []
for n in pages:
    if not os.path.exists(f'work/raw/{n:03d}.json'):
        continue
    res = process_page(n, site)
    crop_and_index(res, rows)
    a, b = res['rng']
    got = {it['num'] for it in res['items']}
    miss = [k for k in res['cands'] if k not in got and k in VALID] if site not in ('X', 'ALL') else []
    lowc = [(it['raw'], it['num'], it['side'], it.get('note', '')) for it in res['items'] if not it['conf']]
    json.dump(res, open(f'work/res_{n:03d}.json', 'w'), default=int)
    diag.append(dict(page=n, rng=f'{a}-{b}', missing=miss, unlinked=len(res['unlinked_photos']), lowconf=lowc, orphans=res['orphan_caps']))
# merge index with other sites already present
idx = 'photos/index.csv'
old = []
if os.path.exists(idx):
    old = [r for r in csv.DictReader(open(idx)) if int(r['page']) not in set(pages)]
fields = ['label', 'side', 'page', 'x', 'y', 'w', 'h', 'caption_raw', 'confident', 'file', 'note']
with open(idx, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fields); w.writeheader()
    for r in old + rows: w.writerow(r)
json.dump(diag, open(f'work/diag_{site}.json', 'w'), default=int, indent=0)
print('rows', len(rows), 'low-conf', sum(1 for r in rows if not r['confident']))
for d in diag:
    if d['missing'] or d['unlinked'] or d['lowconf'] or d['orphans']:
        print(d['page'], d['rng'], 'miss', d['missing'], 'unl', d['unlinked'], 'orph', d['orphans'], 'low', d['lowconf'])
