"""TAHAP 2 - Profiling data mentah: identifikasi elemen, tipe, format, kelengkapan."""
import re

import pandas as pd

from common import OUTPUT_DIR, ensure_dirs, latest_manifest, write_json


def infer_kind(nn):
    if nn.empty:
        return "kosong"
    if pd.to_numeric(nn, errors="coerce").notna().mean() > 0.98:
        return "numerik (ID/angka)"
    if nn.str.match(r"^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}").mean() > 0.9:
        return "tanggal-waktu"
    if nn.nunique() <= 50 or nn.nunique() / len(nn) < 0.05:
        return "kategorikal"
    return "teks bebas"


def date_hint(nn):
    a = b = 0
    for v in nn.head(20000):
        m = re.match(r"^\s*(\d{1,2})[/-](\d{1,2})[/-]", v)
        if m:
            a += int(m.group(1)) > 12
            b += int(m.group(2)) > 12
    return {"bagian1_gt12": a, "bagian2_gt12": b, "contoh": nn.head(3).tolist()}


def main():
    ensure_dirs()
    m = latest_manifest()
    df = pd.read_csv(m["local_csv"], dtype=str, keep_default_na=False)
    rows, hints = [], {}
    for c in df.columns:
        s = df[c].str.strip()
        s = s.mask(s == "")
        nn = s.dropna()
        kind = infer_kind(nn)
        if kind == "tanggal-waktu":
            hints[c] = date_hint(nn)
        top = nn.value_counts().head(5)
        rows.append({
            "kolom": c, "jenis_terdeteksi": kind, "non_null": int(nn.size),
            "null": int(s.isna().sum()), "null_pct": round(float(s.isna().mean()) * 100, 2),
            "unik": int(nn.nunique()), "contoh": nn.iloc[0] if len(nn) else "",
            "top5": "; ".join(f"{k} ({v})" for k, v in top.items()),
        })
    prof = pd.DataFrame(rows)
    summary = {"n_rows": len(df), "n_cols": df.shape[1],
               "exact_duplicate_rows": int(df.duplicated().sum()),
               "columns": rows, "date_hints": hints}
    write_json(OUTPUT_DIR / "profile_raw.json", summary)
    md = (f"# Profil Data Mentah\n\nBaris: {len(df):,} | Kolom: {df.shape[1]} | "
          f"Duplikat persis: {summary['exact_duplicate_rows']:,}\n\n"
          + prof.to_markdown(index=False)
          + "\n\n## Petunjuk format tanggal\n\n" + str(hints) + "\n")
    (OUTPUT_DIR / "profile_raw.md").write_text(md, encoding="utf-8")
    print(prof[["kolom", "jenis_terdeteksi", "null", "unik", "contoh"]].to_string(index=False))
    print(f"\n[OK] outputs/profile_raw.md & profile_raw.json | petunjuk tanggal: {hints}")


if __name__ == "__main__":
    main()
