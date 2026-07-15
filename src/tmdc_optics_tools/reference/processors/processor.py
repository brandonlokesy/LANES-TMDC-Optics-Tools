# import requests

class Processor:
    def __init__(self, meta: dict, out_dir: Path):
        self.meta = meta
        self.out_path = out_dir / f"{meta['material']}__{meta['source']}.h5"

#     def _fetch_zip(self, url: str) -> zipfile.ZipFile:
#         r = requests.get(url, headers=HEADERS)
#         r.raise_for_status()
#         return zipfile.ZipFile(io.BytesIO(r.content))