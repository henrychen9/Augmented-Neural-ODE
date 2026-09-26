# merge_sweeps.py
import torch
import sys

def load(path):
    print(f"loading {path}")
    return torch.load(path, map_location="cpu")

def main():
    if len(sys.argv) < 4:
        print("usage: python merge_sweeps.py out.pt in1.pt in2.pt [in3.pt ...]")
        sys.exit(1)

    out_path = sys.argv[1]
    in_paths = sys.argv[2:]

    all_records = {}
    meta = None

    for p in in_paths:
        obj = load(p)

        if meta is None:
            meta = obj.get("meta", {})

        for r in obj["records"]:
            seed = r["seed"]
            all_records[seed] = r   # later files overwrite earlier ones if duplicated

    # sort by seed
    records = [all_records[s] for s in sorted(all_records.keys())]

    # recompute summary
    good = sum(1 for r in records if r["label"] == "accurate")
    bad  = len(records) - good

    merged = {
        "meta": meta,
        "records": records,
        "summary": {
            "accurate": good,
            "non_accurate_or_failed": bad,
            "accurate_frac": good / max(len(records), 1),
            "total_records": len(records),
        }
    }

    torch.save(merged, out_path)
    print(f"\nmerged {len(records)} total records")
    print(f"saved -> {out_path}")

if __name__ == "__main__":
    main()
