import cv2,subprocess,sys,os
from concurrent.futures import ThreadPoolExecutor
def hdr(n):
    im=cv2.imread(f'pages/p-{n:03d}.pgm',0)
    top=im[:110]
    top=cv2.resize(top,None,fx=3,fy=3,interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(f'/tmp/h{n}.png',top)
    t=subprocess.run(['tesseract',f'/tmp/h{n}.png','-','--psm','6'],capture_output=True,text=True).stdout
    return n,' | '.join(l.strip() for l in t.splitlines() if l.strip())
with ThreadPoolExecutor(4) as ex:
    for n,t in ex.map(hdr,range(int(sys.argv[1]),int(sys.argv[2]))):
        print(n,t)
