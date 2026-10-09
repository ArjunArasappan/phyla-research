from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parent
out=Path(sys.argv[1]);out.mkdir(parents=True,exist_ok=True)
for archive in json.loads((root/'archives.json').read_text())['archives']:
 target=out/archive['name'];temporary=target.with_suffix(target.suffix+'.partial');whole=hashlib.sha256();size=0
 with temporary.open('wb') as f:
  for record in archive['chunks']:
   block=(root/record['path']).read_bytes()
   assert len(block)==record['bytes'] and hashlib.sha256(block).hexdigest()==record['sha256'], record['path']
   f.write(block);whole.update(block);size+=len(block)
 assert size==archive['bytes'] and whole.hexdigest()==archive['sha256'], archive['name']
 temporary.replace(target);print('Verified',target)
