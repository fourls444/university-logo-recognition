"""Bounded official-page collection; downloaded files require manual visual approval."""
import csv,hashlib,io,json,re,sys
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
import requests
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
MISSING=json.loads((ROOT/'data/candidates/still_missing.json').read_text())
def collect(row):
    report={'university':row['name'],'attempts':[]}; found=[]
    for page in [row['url']+'/en',row['url']+'/th',row['url']]:
        try:
            response=requests.get(page,timeout=15);response.raise_for_status()
            urls=re.findall(r'''(?:src|href|content|data-src)\s*=\s*["']([^"']+)["']''',response.text,re.I)
            urls+=re.findall(r'''url\(["']?([^)'" ]+)''',response.text,re.I)
            urls=[urljoin(response.url,u) for u in urls if re.search(r'logo|brand|symbol|ตรา|emblem',u,re.I) and re.search(r'\.(png|jpg|jpeg|webp|gif|svg)(?:\?|$)',u,re.I)]
            report['attempts'].append({'page':response.url,'logo_urls':len(urls)})
            for url in list(dict.fromkeys(urls))[:8]:
                try:
                    r=requests.get(url,timeout=12);r.raise_for_status()
                    raw=r.content
                    if '.svg' in url.lower():
                        import resvg_py
                        raw=resvg_py.svg_to_bytes(svg_string=raw.decode("utf8"),width=512)
                    im=Image.open(io.BytesIO(raw));im.load()
                    if im.width<40 or im.height<30:continue
                    slug=re.sub(r'[^a-z0-9]+','-',row['name'].lower()).strip('-')
                    directory=Path('data/candidates')/slug
                    index=len(list((ROOT/directory).glob('*.png')))+1
                    file=directory/f'{slug}-logo-candidate-{index:02d}.png'
                    (ROOT/file).parent.mkdir(parents=True,exist_ok=True); im.save(ROOT/file)
                    found.append({'university':row['name'],'file':file.as_posix(),'source_page':response.url,'image_url':url,'status':'pending_visual_review','source_group':row['name']+' official branding (keep related variants in one split)','rights':'Public access does not establish redistribution permission'})
                except Exception as e:report['attempts'].append({'image':url,'error':str(e)[:180]})
            if found:break
        except Exception as e:report['attempts'].append({'page':page,'error':str(e)[:180]})
    return found,report
if __name__=='__main__':
    rows=[r for r in csv.DictReader((ROOT/'university_info.csv').open(encoding='utf-8-sig')) if r['name'] in MISSING]
    results=list(ThreadPoolExecutor(max_workers=8).map(collect,rows))
    found=[e for f,r in results for e in f]
    (ROOT/'data/candidates/new_sources.json').write_text(json.dumps(found,indent=2),encoding='utf8')
    (ROOT/'data/candidates/new_collection_report.json').write_text(json.dumps([r for f,r in results],indent=2),encoding='utf8')
    for start in range(0,len(found),20):
        canvas=Image.new('RGB',(1000,((min(20,len(found)-start)+3)//4)*180),'#ddd'); draw=ImageDraw.Draw(canvas)
        from dataset import preprocess_image
        for j,e in enumerate(found[start:start+20]):
            im=preprocess_image(Image.open(ROOT/e['file']));im.thumbnail((240,145))
            x=(j%4)*250;y=(j//4)*180;canvas.paste(im,(x,y));draw.text((x,y+146),str(start+j)+' '+e['university'][:29],fill='black')
        canvas.save(ROOT/f'data/candidates/new_preview_{start//20}.jpg')
    print('Downloaded',len(found),'covering',len({e['university'] for e in found}))
