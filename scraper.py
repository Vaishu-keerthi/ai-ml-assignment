#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrape laptops from WebScraper test site, download images,
and save CSV/JSON with image_path.
Colab usage: from scraper import run_scraper; run_scraper(1, 10, "data/scraped_data.csv", "data/images")
CLI usage:  python scraper.py --start-page 1 --end-page 10 --out data/scraped_data.csv --images-dir data/images
"""
import argparse, time, requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import pandas as pd

BASE_ROOT = "https://webscraper.io"
BASE_URL  = "https://webscraper.io/test-sites/e-commerce/allinone/computers/laptops?page={page}"
HEADERS   = {"User-Agent": "Mozilla/5.0"}

def text_or_none(el): return el.get_text(" ", strip=True) if el else None
def first_from_srcset(srcset: str) -> str:
    if not srcset: return ""
    first = srcset.split(",")[0].strip()
    return first.split()[0] if first else ""
def make_abs(root: str, maybe_url: str) -> str | None:
    if not maybe_url: return None
    url = urljoin(root, maybe_url)
    u = urlparse(url)
    return url if u.scheme in ("http","https") and u.netloc else None

def scrape_pages(start_page: int, end_page: int) -> pd.DataFrame:
    rows = []
    for p in range(start_page, end_page + 1):
        url = BASE_URL.format(page=p)
        r = requests.get(url, headers=HEADERS, timeout=30); r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for card in soup.select(".thumbnail"):
            title_a = card.select_one(".title")
            price_el = card.select_one(".price")
            desc_el  = card.select_one(".description")
            reviews_el = card.select_one(".ratings .pull-right")
            img = card.select_one("img")
            rows.append({
                "title": text_or_none(title_a),
                "price": text_or_none(price_el),
                "description": text_or_none(desc_el),
                "reviews_text": text_or_none(reviews_el),
                "img_src": img.get("src") if img else None,
                "img_srcset": img.get("srcset") if img else None,
                "page_url": url
            })
        print(f"[page {p}] total rows so far: {len(rows)}"); time.sleep(0.2)
    df = pd.DataFrame(rows)
    if "reviews_text" in df.columns:
        df["reviews_count"] = df["reviews_text"].fillna("").str.extract(r"(\d+)").astype(float)
    return df

def download_images(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, r in df.iterrows():
        raw = r.get("img_srcset") or r.get("img_src")
        if not raw: paths.append(None); continue
        if r.get("img_srcset"): raw = first_from_srcset(raw)
        url = make_abs(BASE_ROOT, raw)
        if not url: paths.append(None); continue
        f = out_dir / f"img_{i:06d}.jpg"
        if f.exists(): paths.append(str(f)); continue
        try:
            rr = requests.get(url, headers=HEADERS, timeout=30); rr.raise_for_status()
            f.write_bytes(rr.content); paths.append(str(f)); time.sleep(0.1)
        except Exception: paths.append(None)
    df["image_path"] = paths
    return df

def run_scraper(start_page=1, end_page=10, out_csv="data/scraped_data.csv", images_dir="data/images"):
    Path(images_dir).mkdir(parents=True, exist_ok=True)
    df = scrape_pages(start_page, end_page)
    df = download_images(df, Path(images_dir))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    df.to_json(out_csv.replace(".csv", ".json"), orient="records")
    print(f"[DONE] rows={len(df)} → {out_csv} | images in {images_dir}")
    return df

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-page", type=int, default=1)
    ap.add_argument("--end-page",   type=int, default=10)
    ap.add_argument("--out",        default="data/scraped_data.csv")
    ap.add_argument("--images-dir", default="data/images")
    args, _ = ap.parse_known_args()  # safe in Colab
    run_scraper(args.start_page, args.end_page, args.out, args.images_dir)
