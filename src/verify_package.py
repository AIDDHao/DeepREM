"""Check distributed files against SHA-256 manifest."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
items=json.loads((root/'files.sha256.json').read_text())
for path,expected in items.items():
 actual=hashlib.sha256((root/path).read_bytes()).hexdigest()
 assert actual==expected,path
print(f'PASS: {len(items)} file checksums')
