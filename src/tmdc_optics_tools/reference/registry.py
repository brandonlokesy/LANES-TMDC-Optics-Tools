from .processors.Tagarelli2023 import Tagarelli2023Processor

REGISTRY = [
    {
        "material":  "WSe2_bilayer",
        "source":    "Tagarelli2023",
        "zenodo_doi": "10.5281/zenodo.7660668",
        "title":     "Electrical control of hybrid exciton transport in a van der Waals heterostructure",
        "doi" : "10.1038/s41566-023-01198-w",
        "comment" : "",
        "processor": Tagarelli2023Processor,
    },
]

if __name__ == "__main__":
    from pathlib import Path
    Path("reference_data/data").mkdir(exist_ok=True)

    for entry in REGISTRY:
        print(f"Processing {entry['source']} ({entry['material']})...")
        proc = entry["processor"](entry)
        proc.run()
        print(f"  → Done\n")