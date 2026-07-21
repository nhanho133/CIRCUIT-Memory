"""
calibrate_conf.py — hiệu chỉnh confidence của cue-extractor (Cách 5).

Nhãn đúng/sai mỗi trục = so giá trị LLM trích với ORACLE cue (qa["cue_completed"]).
Fit MỘT Isotonic Regression RIÊNG cho từng trục (time/spaces/entities/content) — đúng
khớp Method §3.2 "fit an isotonic regression g_k per axis" (bản cũ fit gộp 1 calibrator
chung cho cả 4 trục — đã sửa).

Held-out thật sự cho MỌI câu trong tập báo cáo (không chỉ test-122):
  - test-122 (--sample) luôn bị loại khỏi mọi lần fit, dự đoán bằng calibrator fit trên
    toàn bộ DEV còn lại (546 câu) — giữ nguyên như bản cũ, đã held-out đúng nghĩa.
  - DEV (546 câu) trước đây được calibrator CHÍNH NÓ fit rồi tự áp lên chính nó (rò rỉ
    nhãn). Nay dùng K-fold cross-fitting (mặc định K=5) NỘI BỘ trong DEV: mỗi câu trong
    DEV nhận confidence từ 1 calibrator chỉ fit trên (K-1)/K phần DEV còn lại, không bao
    giờ thấy nhãn của chính câu đó.

Báo cáo ECE trước/sau, theo TỪNG trục. Ghi extracted_cues_calibrated.json.

Usage:
  python calibrate_conf.py --base data_root_ep200 --book_id 1 \
      --sample data_root_sample/book1/qa_book1.json --kfold 5
"""
import os, re, json, argparse, pickle
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

AXES = ["time", "spaces", "entities", "content"]
SLOT2AXIS = {"t": "time", "s": "spaces", "l": "spaces", "ent": "entities",
             "e": "entities", "c": "content"}


def parse_oracle(cue, cue_completed):
    """oracle cue_completed '(*, *, {Dr. Vega}, *)' + cue '(*,*,ent,*)' -> {axis: value}"""
    slots = [s.strip().lower().strip("{}") for s in cue.strip("()").split(",")]
    toks = re.findall(r"\{([^}]+)\}", cue_completed)
    out, ti = {}, 0
    for s in slots:
        if s == "*":
            continue
        ax = SLOT2AXIS.get(s)
        val = toks[ti] if ti < len(toks) else None
        ti += 1
        if ax and val:
            out[ax] = val.strip()
    return out


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()


def is_correct(extracted_val, oracle_val):
    a, b = norm(extracted_val), norm(oracle_val)
    return a == b or (len(a) > 3 and (a in b or b in a))


def ece(confs, labels, bins=10):
    confs, labels = np.asarray(confs), np.asarray(labels)
    if len(confs) == 0:
        return float("nan")
    e, n = 0.0, len(confs)
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        hi = lo + 1.0 / bins
        m = (confs >= lo) & (confs < hi if hi < 1 else confs <= hi)
        if m.sum():
            e += m.sum() / n * abs(confs[m].mean() - labels[m].mean())
    return e


def fit_isotonic(c, y):
    cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    cal.fit(c, y)
    return cal


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data_root_ep200")
    ap.add_argument("--book_id", type=int, default=1)
    ap.add_argument("--sample", default="data_root_sample/book1/qa_book1.json",
                    help="qa của tập TEST (122) — để loại khỏi mọi lần fit")
    ap.add_argument("--kfold", type=int, default=5,
                    help="số fold cross-fitting NỘI BỘ trong DEV (mỗi câu DEV được dự "
                         "đoán bởi calibrator không thấy nhãn của chính nó)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--apply_only", default=None,
                    help="path tới calibrator.pkl đã fit ở SÁCH KHÁC (vd. Uscifi) — nếu set, "
                         "KHÔNG fit lại, chỉ áp calibrator có sẵn lên sách này (test tổng quát "
                         "hóa xuyên sách, đúng nghĩa cross-book validation).")
    args = ap.parse_args()

    bk = f"{args.base}/book{args.book_id}"
    qa = json.load(open(f"{bk}/qa_book{args.book_id}.json"))
    cues = json.load(open(f"{bk}/extracted_cues.json"))
    test_qs = {q["question"] for q in json.load(open(args.sample))} if os.path.exists(args.sample) else set()

    # gom điểm dữ liệu PER-AXIS: {axis: [(q_idx, conf_raw, correct, is_test), ...]}
    pts_by_axis = {a: [] for a in AXES}
    axis_stat = {a: [0, 0] for a in AXES}
    for k, cue in cues.items():
        i = int(k)
        oracle = parse_oracle(qa[i]["cue"], qa[i]["cue_completed"])
        is_test = qa[i]["question"] in test_qs
        for a in AXES:
            ex = cue.get(a)
            if not ex:
                continue
            ok = 1 if (a in oracle and is_correct(ex["value"], oracle[a])) else 0
            axis_stat[a][0] += ok
            axis_stat[a][1] += 1
            pts_by_axis[a].append((i, ex["confidence"], ok, is_test))

    print(f"Điểm dữ liệu trích theo trục: " + ", ".join(f"{a}={len(pts_by_axis[a])}" for a in AXES))
    print("Độ chính xác trích theo trục (đúng/tổng):")
    for a in AXES:
        d, t = axis_stat[a]
        print(f"  {a:9s}: {d}/{t}" + (f" = {d/t:.2f}" if t else ""))

    if args.apply_only:
        print(f"\n[apply_only] Tái sử dụng calibrator từ {args.apply_only} (KHÔNG fit lại trên sách này)")
        transferred = pickle.load(open(args.apply_only, "rb"))
        out = {}
        for a in AXES:
            rows = pts_by_axis[a]
            raw = np.array([p[1] for p in rows]); y = np.array([p[2] for p in rows])
            cal = transferred.get(a)
            pred = cal.predict(raw) if (cal is not None and len(rows)) else raw
            if len(rows):
                print(f"  [{a}] n={len(rows)} ECE trước={ece(raw, y):.4f}  "
                      f"ECE sau (calibrator transferred)={ece(pred, y):.4f}")
        calibrated_by_qidx_axis = {}
        for a in AXES:
            rows = pts_by_axis[a]
            cal = transferred.get(a)
            for (i, c, y, _) in rows:
                calibrated_by_qidx_axis[(i, a)] = float(cal.predict([c])[0]) if cal is not None else c
        out = {}
        for k, cue in cues.items():
            i = int(k)
            nc = {}
            for a in AXES:
                ex = cue.get(a)
                nc[a] = None if not ex else {
                    "value": ex["value"],
                    "confidence": calibrated_by_qidx_axis.get((i, a), ex["confidence"]),
                    "confidence_raw": ex["confidence"],
                }
            out[k] = nc
        json.dump(out, open(f"{bk}/extracted_cues_calibrated.json", "w"), ensure_ascii=False, indent=1)
        print(f"\nĐã ghi {bk}/extracted_cues_calibrated.json (calibrator transferred, cross-book)")
        raise SystemExit(0)

    # ---- per-axis: fit calibrator riêng, k-fold cross-fit trong DEV, held-out cho TEST
    per_axis_cal_full = {}       # calibrator fit trên TOÀN BỘ DEV của trục đó (dùng cho TEST + final output)
    calibrated_by_qidx_axis = {}  # (q_idx, axis) -> calibrated confidence (đúng nghĩa held-out cho ECE)
    rng = np.random.RandomState(args.seed)

    for a in AXES:
        pts = pts_by_axis[a]
        dev_pts = [p for p in pts if not p[3]]
        test_pts = [p for p in pts if p[3]]
        raw_a = np.array([p[1] for p in pts]); y_a = np.array([p[2] for p in pts])
        print(f"\n[{a}] raw conf: mean={raw_a.mean():.3f} std={raw_a.std():.3f} | "
              f"ECE trước calibrate = {ece(raw_a, y_a):.4f}  (n={len(pts)}, DEV={len(dev_pts)}, TEST={len(test_pts)})")

        if len(dev_pts) < args.kfold or len(set(p[2] for p in dev_pts)) < 2:
            print(f"  CẢNH BÁO [{a}]: DEV không đủ đa dạng/số lượng → bỏ qua calibration trục này.")
            for (i, c, y, _) in pts:
                calibrated_by_qidx_axis[(i, a)] = c
            per_axis_cal_full[a] = None
            continue

        dev_c = np.array([p[1] for p in dev_pts]); dev_y = np.array([p[2] for p in dev_pts])
        dev_idx = np.array([p[0] for p in dev_pts])

        # k-fold cross-fitting NỘI BỘ DEV: mỗi điểm DEV dự đoán bởi calibrator không thấy nó
        kf = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
        held_out_pred = np.zeros(len(dev_pts))
        for tr_idx, te_idx in kf.split(dev_c):
            fold_cal = fit_isotonic(dev_c[tr_idx], dev_y[tr_idx])
            held_out_pred[te_idx] = fold_cal.predict(dev_c[te_idx])
        for j, (i, c, y, _) in enumerate(dev_pts):
            calibrated_by_qidx_axis[(i, a)] = float(held_out_pred[j])

        # calibrator fit trên TOÀN BỘ DEV — dùng cho TEST-122 (đã held-out sẵn theo thiết kế gốc)
        full_cal = fit_isotonic(dev_c, dev_y)
        per_axis_cal_full[a] = full_cal
        for (i, c, y, _) in test_pts:
            calibrated_by_qidx_axis[(i, a)] = float(full_cal.predict([c])[0])

        cal_a_all = np.array([calibrated_by_qidx_axis[(p[0], a)] for p in pts])
        print(f"  ECE sau calibrate  = {ece(cal_a_all, y_a):.4f}  (held-out: DEV qua {args.kfold}-fold, TEST qua full-DEV-calibrator)")

    pickle.dump(per_axis_cal_full, open(f"{bk}/calibrator.pkl", "wb"))

    # ---- áp calibrated confidence (per-axis, held-out) → cues_calibrated
    out = {}
    for k, cue in cues.items():
        i = int(k)
        nc = {}
        for a in AXES:
            ex = cue.get(a)
            if not ex:
                nc[a] = None
            else:
                c = calibrated_by_qidx_axis.get((i, a), ex["confidence"])
                nc[a] = {"value": ex["value"], "confidence": c, "confidence_raw": ex["confidence"]}
        out[k] = nc
    json.dump(out, open(f"{bk}/extracted_cues_calibrated.json", "w"), ensure_ascii=False, indent=1)
    print(f"\nĐã ghi {bk}/extracted_cues_calibrated.json (per-axis, {args.kfold}-fold held-out cho DEV)")
