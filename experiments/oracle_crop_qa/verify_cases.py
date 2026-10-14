#!/usr/bin/env python3
"""Check the case browser (cases/) against the run results, and render it in a headless browser.

  --stage data     (worldmm venv: decord, PIL) the page's data vs run_data/results; tab membership recomputed
                   independently; for a sample of questions, rebuild every condition's QA input with oracle_crop.py
                   and compare its image hash with the one stored at QA time (= the model saw exactly these pixels)
  --stage browser  (a venv with playwright) open cases/index.html from disk: JS errors, tab/filter counts,
                   broken images, screenshots in cases_check/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDS = ("full", "gaze", "center", "dual_center", "dual_gaze", "dual_offcenter", "dual_offcenter_center")


def load_page_data(cases: Path) -> dict:
    text = (cases / "data.js").read_text(encoding="utf-8")
    return json.loads(text[len("window.DATA = "):].rstrip().rstrip(";"))


def check(ok: bool, msg: str, fails: list) -> None:
    print(("PASS " if ok else "FAIL ") + msg)
    if not ok:
        fails.append(msg)


def stage_data(args) -> list:
    sys.path.insert(0, str(HERE))
    import oracle_crop as oc
    from gaze_common import load_gaze

    fails: list = []
    data = load_page_data(args.cases)
    run = args.run
    qids = oc.base.read_json(run / "crop_manifest.json")["question_ids"]
    res = {c: {q: oc.base.read_json(run / "results" / c / f"{q}.json") for q in qids} for c in CONDS}
    check(all(res[c][q]["status"] == "ok" for c in CONDS for q in qids), f"all {len(qids)}x{len(CONDS)} results ok", fails)

    # tab membership, recomputed without the page builder's code
    corr = {q for q in qids if len({bool(res[c][q]["correct"]) for c in CONDS}) > 1}
    norm = lambda r: (r or "").strip().lstrip("(").upper()[:1]
    lett = {q for q in qids if len({norm(res[c][q]["response"]) for c in CONDS}) > 1}
    page = {c["id"]: c for c in data["cases"]}
    check({q for q, c in page.items() if "correct" in c["tabs"]} == corr, f"tab 'correct' = {len(corr)} questions", fails)
    check({q for q, c in page.items() if "letter" in c["tabs"]} == lett, f"tab 'letter' = {len(lett)} questions", fails)
    check(corr <= lett, "every correctness change is also a letter change", fails)
    groups = {g: sum(1 for q in corr if page[q]["group"] == g) for g in ("center", "full", "other")}
    want_c = sum(1 for q in qids if res["center"][q]["correct"] and not res["full"][q]["correct"])
    want_f = sum(1 for q in qids if res["full"][q]["correct"] and not res["center"][q]["correct"])
    check(groups["center"] == want_c and groups["full"] == want_f, f"groups in tab 1: {groups} (center O/full X {want_c}, "
          f"full O/center X {want_f})", fails)

    # every case: replies, correctness, image counts, files, boxes
    bad = []
    for q, c in page.items():
        for cond in CONDS:
            r, p = res[cond][q], c["conds"][cond]
            if p["response"] != r["response"] or p["correct"] != bool(r["correct"]) or p["n_img"] != r["num_images"]:
                bad.append(f"{q}/{cond}")
    check(not bad, f"replies / correctness / image counts match results for {len(page)} cases" + (f" ({bad[:5]})" if bad else ""), fails)
    missing = [f["img"][v] for c in page.values() for f in c["frames"] for v in ("full", "center", "gaze")
               if not (args.cases / f["img"][v]).exists()]
    check(not missing, f"all {sum(len(c['frames']) * 3 for c in page.values())} image files exist", fails)
    out_box = [c["id"] for c in page.values() for f in c["frames"] for b in f["boxes"].values()
               if not (0 <= b[0] and 0 <= b[1] and b[0] + b[2] <= 1.0001 and b[1] + b[3] <= 1.0001)]
    check(not out_box, "all crop boxes lie inside the frame", fails)
    frames_ok = []
    for q, c in page.items():
        info = oc.base.read_json(run / "frames" / f"{q}.json")
        frames_ok.append(all(info[f["i"]]["gaze"] == f["gaze"] and info[f["i"]]["off"] == f["off"] for f in c["frames"]))
    check(all(frames_ok), "shown gaze points / off-center flags match run_data/frames", fails)
    check(sum(1 for c in page.values() if any(f["target"] for f in c["frames"])) == len(page),
          "every case shows its target-nearest timestamp", fails)

    # the model's input, rebuilt: hash must equal the hash recorded at QA time
    clips = {k["key"]: k for k in oc.base.read_json(oc.base.ROOT / "experiments/gaze_crop/pool_all.json")["clips"]}
    gaze = load_gaze(str(oc.base.ROOT / "experiments/gaze_crop/gaze_points.json"))
    sample = sorted(random.Random(0).sample(sorted(page), args.n_hash), key=int)
    mism = []
    for q in sample:
        slots = oc.timestamps(res["full"][q]["target_keys"], clips, Path(oc.VIDEO_ROOT), decode=True)
        info, frames = oc.frame_info(slots, gaze), oc.decode(slots)
        for cond in CONDS:
            _, ids, h = oc.build_input(cond, frames, info)
            if h != res[cond][q]["image_hash"] or ids != res[cond][q]["frame_ids"]:
                mism.append(f"{q}/{cond}")
        # the page's crops come from the same crop_box: compare one shown center/gaze crop with the QA crop
        f = page[q]["frames"][0]
        from PIL import Image, ImageChops
        for view in ("center", "gaze"):
            qa = oc.view_image(frames[f["i"]], view, f["gaze"]).copy()
            qa.thumbnail((256, 256))
            shown = Image.open(args.cases / f["img"][view]).convert("RGB")
            diff = sum(ImageChops.difference(qa, shown).convert("L").histogram()[i] * i for i in range(256)) / (qa.width * qa.height)
            if qa.size != shown.size or diff > 6:
                mism.append(f"{q}/{view}-thumb diff={diff:.1f}")
    check(not mism, f"QA inputs rebuilt for {len(sample)} cases x 7 conditions: image hashes equal the stored ones; "
          f"shown crops match the QA crops" + (f" ({mism[:6]})" if mism else ""), fails)
    return fails


def stage_browser(args) -> list:
    from playwright.sync_api import sync_playwright

    fails: list = []
    data = load_page_data(args.cases)
    shots = HERE / "cases_check"
    shots.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1400, "height": 1000})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto((args.cases / "index.html").resolve().as_uri())
        page.wait_for_selector(".card")
        tabs = page.locator(".tab").all_inner_texts()
        check(tabs == [f"정답 여부 변화 ({data['counts']['correct']})", f"보기(A~D) 변화 ({data['counts']['letter']})"],
              f"tab labels {tabs}", fails)
        n1 = page.locator(".card").count()
        check(n1 == data["counts"]["correct"], f"tab 1 renders {n1} cards", fails)
        page.locator(".card").first.screenshot(path=str(shots / "tab1_first_card.png"))
        page.select_option("#f-group", "center")
        check(page.locator(".card").count() == sum(1 for c in data["cases"] if "correct" in c["tabs"] and c["group"] == "center"),
              f"group filter 'center O · full X' -> {page.locator('.card').count()} cards", fails)
        page.select_option("#f-group", "full")
        check(page.locator(".card").count() == sum(1 for c in data["cases"] if "correct" in c["tabs"] and c["group"] == "full"),
              f"group filter 'full O · center X' -> {page.locator('.card').count()} cards", fails)
        page.select_option("#f-group", "")
        page.select_option("#f-audio", "0")
        nv = page.locator(".card").count()
        check(nv == sum(1 for c in data["cases"] if "correct" in c["tabs"] and not c["need_audio"]), f"evidence filter -> {nv} cards", fails)
        page.select_option("#f-audio", "")
        some_id = next(c["id"] for c in data["cases"] if "correct" in c["tabs"])
        page.fill("#f-id", some_id)
        check(page.locator(".card").count() == 1, f"ID filter '{some_id}' -> 1 card", fails)
        page.fill("#f-id", "")
        page.click(".tab[data-tab=letter]")
        n2 = page.locator(".card").count()
        check(n2 == data["counts"]["letter"], f"tab 2 renders {n2} cards", fails)
        # load every image (lazy -> eager) and count broken ones
        page.evaluate("document.querySelectorAll('img').forEach(i => i.loading = 'eager')")
        page.wait_for_function("[...document.images].every(i => i.complete)", timeout=120000)
        n_img = page.evaluate("document.images.length")
        broken = page.evaluate("[...document.images].filter(i => !i.naturalWidth).map(i => i.getAttribute('src'))")
        check(not broken, f"{n_img} images loaded, broken: {len(broken)}" + (f" {broken[:3]}" if broken else ""), fails)
        page.select_option("#f-group", "center")
        page.locator(".card").first.screenshot(path=str(shots / "tab2_center_first_card.png"))
        page.set_viewport_size({"width": 390, "height": 900})
        overflow = page.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
        check(not overflow, "no horizontal page scroll at phone width (frame strip scrolls inside the card)", fails)
        page.screenshot(path=str(shots / "phone.png"))
        check(not errors, f"no JS errors ({errors[:3]})" if errors else "no JS errors", fails)
        browser.close()
    print(f"screenshots: {shots}")
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("data", "browser"), required=True)
    ap.add_argument("--cases", type=Path, default=HERE / "cases")
    ap.add_argument("--run", type=Path, default=HERE / "run_data")
    ap.add_argument("--n-hash", type=int, default=10, help="cases whose QA input is rebuilt and hash-checked")
    args = ap.parse_args()
    fails = stage_data(args) if args.stage == "data" else stage_browser(args)
    print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAIL'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
