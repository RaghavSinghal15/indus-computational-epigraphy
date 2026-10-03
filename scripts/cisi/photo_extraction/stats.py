"""Summary statistics of photos/index.csv."""
import csv, collections, json, glob
rows = list(csv.DictReader(open('photos/index.csv')))
valid = [r['cisi_label'] for r in csv.DictReader(open('cisi_im77.csv'))]
site = lambda l: l.split('-')[0]
print('total photos', len(rows), 'flagged (confident=0)', sum(r['confident'] == '0' for r in rows))
per = collections.Counter(site(r['label']) for r in rows)
lab = collections.defaultdict(set)
for r in rows: lab[r['label']].add(r['side'])
for s, c in per.most_common():
    vs = [l for l in valid if site(l) == s]
    got = [l for l in vs if l in lab]
    print(f'{s:5s} photos {c:5d}  labels covered {len(got)}/{len(vs)}  flagged {sum(r["confident"]=="0" for r in rows if site(r["label"])==s)}')
m = [l for l in valid if site(l) == 'M']
imp = [l for l in m if any(sd[:1].islower() for sd in lab.get(l, ()))]
obj = [l for l in m if l in lab and l not in imp]
print('M labels with impression photo (lowercase side):', len(imp))
print('M labels with only object photo(s):', len(obj))
print('M labels with no photo:', [l for l in m if l not in lab])
print('labels not in list:', sorted({r['label'] for r in rows if r['label'] not in valid}))
