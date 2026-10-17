# -*- coding: utf-8 -*-
"""
🏇 自分専用 競馬予想AI（v5）× 展開・血統

必要ファイル（GitHubリポジトリ直下）:
  - app.py / requirements.txt / model.pkl

Streamlit Cloud の Secrets（Settings > Secrets）に以下を設定:
  [connections.gsheets]
  spreadsheet = "https://docs.google.com/spreadsheets/d/xxxxxxxx/edit"
  type = "service_account"
  project_id = "..."
  private_key_id = "..."
  private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
  client_email = "xxx@xxx.iam.gserviceaccount.com"
  client_id = "..."
  auth_uri = "https://accounts.google.com/o/oauth2/auth"
  token_uri = "https://oauth2.googleapis.com/token"
  auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
  client_x509_cert_url = "..."

スプレッドシートに "predictions" シートを作り、サービスアカウントのメールを「編集者」で共有してください。
"""
import csv
import pickle
import re
import zlib
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_gsheets import GSheetsConnection

st.set_page_config(page_title="競馬予想AI v5", page_icon="🏇", layout="centered")
st.title("🏇 競馬予想AI v5")

MODEL_PATH = Path(__file__).parent / "model.pkl"
JST = timezone(timedelta(hours=9))

STD_COLS = [
    "Bracket", "Horse Number", "Horse Name", "Sex", "Horse Age",
    "Burden Weight", "Jockey", "Trainer", "Sire", "Tactics",
    "Total Prize", "Last Placing", "Last Time",
]
NUM_DEFAULTS = {  # 数値列: 欠損時の補完方法（Tactics は別処理）
    "Bracket": 0, "Horse Number": 0, "Horse Age": 4, "Burden Weight": 55,
    "Total Prize": 0, "Last Placing": "median:9", "Last Time": "mean:90",
    "Distance": 0,
}
STR_COLS = ["Horse Name", "Sex", "Jockey", "Trainer", "Sire",
            "Venue", "Track Condition", "Weather"]
CAT_COLS = ["Venue", "Track Condition", "Weather", "Jockey", "Trainer", "Sire", "Sex"]
TACTICS_LABEL = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}

VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉", "その他"]
DISTANCES = sorted(set(list(range(1000, 3700, 200)) + [1150, 1700, 1800, 2100, 2300, 2500, 2600, 2800, 3000]))
TRACKS = ["良", "稍重", "重", "不良"]
WEATHERS = ["晴", "曇", "小雨", "雨", "雪"]


def norm(s) -> str:
    return re.sub(r"[\s_\-]", "", str(s)).lower()


# ------------------------------------------------------------------
# 数値化ユーティリティ（pandasの文字列型に依存しない）
# ------------------------------------------------------------------
def _is_missing(v) -> bool:
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def num_scalar(v) -> float:
    """任意の値から最初の数値を取り出して float にする。取れなければ NaN。"""
    if _is_missing(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    m = re.search(r"-?\d+(?:\.\d+)?", str(v).replace(",", ""))
    return float(m.group()) if m else np.nan


def to_float_array(values) -> np.ndarray:
    return np.array([num_scalar(v) for v in values], dtype=float)


def safe_stat(arr: np.ndarray, kind: str) -> float:
    """NaN を除いた median / mean。全部 NaN なら NaN を返す。"""
    finite = arr[~np.isnan(arr)]
    if finite.size == 0:
        return np.nan
    return float(np.median(finite) if kind == "median" else np.mean(finite))


# ------------------------------------------------------------------
# モデル読み込み
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="AIの頭脳を読み込み中…")
def load_model():
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)

    meta = {"features": None, "encoders": {}}
    model = obj

    if isinstance(obj, dict):
        model = None
        for k in ("model", "pipeline", "clf", "estimator", "booster", "lgbm", "xgb"):
            if k in obj:
                model = obj[k]
                break
        if model is None:
            model = next((v for v in obj.values() if hasattr(v, "predict")), None)
        for k in ("features", "feature_names", "feature_cols", "feature_columns", "columns"):
            if k in obj and obj[k] is not None:
                meta["features"] = list(obj[k])
                break
        for k in ("encoders", "label_encoders", "encoder", "le_dict"):
            if k in obj and isinstance(obj[k], dict):
                meta["encoders"] = obj[k]
                break
    elif isinstance(obj, (tuple, list)):
        model = next((v for v in obj if hasattr(v, "predict")), None)
        for v in obj:
            if isinstance(v, dict) and not meta["encoders"]:
                meta["encoders"] = v
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
                meta["features"] = list(v)

    if model is None or not hasattr(model, "predict"):
        raise ValueError("model.pkl の中に predict() を持つモデルが見つかりませんでした。")
    return model, meta


def get_feature_names(model, meta):
    if meta.get("features"):
        return meta["features"]
    for attr in ("feature_names_in_", "feature_name_", "feature_names_", "feature_names"):
        v = getattr(model, attr, None)
        if v is not None and not callable(v):
            try:
                return list(v)
            except TypeError:
                pass
    booster = getattr(model, "booster_", None)
    if booster is not None:
        try:
            return list(booster.feature_name())
        except Exception:
            pass
    if hasattr(model, "get_booster"):
        try:
            names = model.get_booster().feature_names
            if names:
                return list(names)
        except Exception:
            pass
    if hasattr(model, "feature_name"):
        try:
            return list(model.feature_name())
        except Exception:
            pass
    return None


# ------------------------------------------------------------------
# 入力データの解析・補正
# ------------------------------------------------------------------
def parse_csv(text: str) -> pd.DataFrame:
    text = text.replace("```csv", "").replace("```", "").replace("\u3000", " ").strip()
    if not text:
        raise ValueError("出馬表が空です。")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    sep = "\t" if "\t" in lines[0] else ","
    rows = [[c.strip() for c in r]
            for r in csv.reader(lines, delimiter=sep, skipinitialspace=True)]

    # ヘッダー判定
    std_lookup = {norm(s): s for s in STD_COLS}
    hits = [std_lookup.get(norm(c)) for c in rows[0]]
    if sum(h is not None for h in hits) >= 8:
        header = [h if h else f"_extra{i}" for i, h in enumerate(hits)]
        rows = rows[1:]
    else:
        if len(rows[0]) < len(STD_COLS):
            raise ValueError("列数が足りません。13列（Bracket〜Last Time）のCSVを貼り付けてください。")
        header = list(STD_COLS)

    ncols = len(header)
    tp_idx = header.index("Total Prize") if "Total Prize" in header else None
    fixed = []
    for r in rows:
        r = list(r)
        # 賞金の「1,234」のようなカンマ区切りで列が増えた行を結合して戻す
        if len(r) > ncols and tp_idx is not None:
            extra = len(r) - ncols
            r = r[:tp_idx] + ["".join(r[tp_idx: tp_idx + extra + 1])] + r[tp_idx + extra + 1:]
        r = (r + [None] * ncols)[:ncols]
        fixed.append(r)

    if not fixed:
        raise ValueError("出馬表のデータ行が読み取れませんでした（ヘッダーのみ、または空です）。")

    df = pd.DataFrame(fixed, columns=header, dtype="object")
    for c in STD_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[STD_COLS].copy()

    # 全列を「文字列 or None」の object 型に統一
    for c in df.columns:
        vals = []
        for v in df[c]:
            if _is_missing(v):
                vals.append(None)
                continue
            s = str(v).strip()
            vals.append(None if s in ("", "nan", "None", "<NA>") else s)
        df[c] = pd.Series(vals, index=df.index, dtype="object")

    df = df.dropna(how="all")
    df = df[~(df["Horse Name"].isna() & df["Horse Number"].isna())]
    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("出馬表から1頭も読み取れませんでした。馬名・馬番の列を確認してください。")
    return df


def parse_time(v):
    if _is_missing(v):
        return np.nan
    s = str(v).strip()
    m = re.match(r"^(\d+):(\d+(?:\.\d+)?)$", s)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.match(r"^(\d+)[.:](\d{2})[.:](\d+)$", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + float("0." + m.group(3))
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_tactics(v):
    """脚質を 1=逃げ 2=先行 3=差し 4=追込 に正規化。"""
    if _is_missing(v):
        return np.nan
    s = str(v).strip()
    m = re.match(r"^([1-4])(?:\.0+)?$", s)
    if m:
        return float(m.group(1))
    for key, code in (("逃", 1), ("先", 2), ("差", 3), ("追", 4)):
        if key in s:
            return float(code)
    return np.nan


def preprocess(df: pd.DataFrame, venue, distance, track, weather) -> pd.DataFrame:
    df = df.copy()
    df["Venue"] = venue
    df["Distance"] = distance
    df["Track Condition"] = track
    df["Weather"] = weather

    # 数値列: 補完（平均・中央値）はせず、読めない・空の値は 0 として扱う
    for c in NUM_DEFAULTS:
        if c == "Last Time":
            arr = np.array([parse_time(v) for v in df[c]], dtype=float)
        else:
            arr = to_float_array(df[c])
        df[c] = pd.Series(np.nan_to_num(arr, nan=0.0), index=df.index, dtype="float64")

    # 脚質: 表示用ラベルと、モデル入力用の数値
    code_arr = np.array([parse_tactics(v) for v in df["Tactics"]], dtype=float)
    df["TacticsCode"] = pd.Series(code_arr, index=df.index, dtype="float64")
    df["脚質ラベル"] = [
        TACTICS_LABEL.get(int(x), "不明") if not np.isnan(x) else "不明" for x in code_arr
    ]
    df["Tactics"] = pd.Series(np.nan_to_num(code_arr, nan=0.0), index=df.index, dtype="float64")

    for c in STR_COLS:
        if c in df.columns:
            df[c] = pd.Series(
                ["不明" if _is_missing(v) else str(v).strip() for v in df[c]],
                index=df.index, dtype="object",
            )

    # 馬番が取れなかった場合は行順で補完
    if (df["Horse Number"] == 0).all():
        df["Horse Number"] = np.arange(1, len(df) + 1, dtype=float)

    return df


# ------------------------------------------------------------------
# 展開予想・ペース診断
# ------------------------------------------------------------------
def diagnose_pace(df: pd.DataFrame) -> dict:
    code = df["TacticsCode"].dropna()
    known = len(code)
    total = len(df)
    if known == 0:
        return {"ok": False, "total": total}

    n_nige = int((code == 1).sum())
    n_senko = int((code == 2).sum())
    n_sashi = int((code == 3).sum())
    n_oikomi = int((code == 4).sum())
    front_ratio = (n_nige + n_senko) / known

    z = 10.0 * (front_ratio - 0.42) + 0.9 * (n_nige - 1.5)
    high_prob = float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.01, 0.99)) * 100

    if high_prob >= 60:
        pace = "ハイペース"
    elif high_prob <= 35:
        pace = "スローペース"
    else:
        pace = "ミドルペース"

    return {
        "ok": True, "pace": pace, "high_prob": round(high_prob, 1),
        "front_ratio": round(front_ratio * 100, 1),
        "n_nige": n_nige, "n_senko": n_senko, "n_sashi": n_sashi, "n_oikomi": n_oikomi,
        "known": known, "total": total,
    }


def render_pace(p: dict, out: pd.DataFrame):
    st.subheader("③ 展開予想・ペース診断")
    if not p.get("ok"):
        st.info("脚質（Tactics）が読み取れなかったため、ペース診断は行えませんでした。")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("想定ペース", p["pace"])
    m2.metric("ハイペース確率", f"{p['high_prob']:.1f}%")
    m3.metric("逃げ・先行の割合", f"{p['front_ratio']:.1f}%")

    breakdown = (f"逃げ{p['n_nige']}頭 / 先行{p['n_senko']}頭 / "
                 f"差し{p['n_sashi']}頭 / 追込{p['n_oikomi']}頭")

    if p["pace"] == "ハイペース":
        st.warning(f"🔥 **ハイペース想定**（ハイペース確率 {p['high_prob']:.1f}%）\n\n"
                   "逃げ・先行馬が多く、前が競り合って流れが速くなりそうです。"
                   "**差し馬が有利になりそう**で、前に行く馬は最後に脚が鈍るリスクがあります。\n\n"
                   f"脚質内訳: {breakdown}")
        favored, tag = ["差し", "追込"], "差し・追込"
    elif p["pace"] == "スローペース":
        st.info(f"🐢 **スローペース想定**（ハイペース確率 {p['high_prob']:.1f}%）\n\n"
                "前に行きたい馬が少なく、流れが落ち着きそうです。"
                "**前残りの展開になりそう**で、逃げ・先行馬が粘り込みやすい一方、"
                "後方の馬は展開面で不利になりがちです。\n\n"
                f"脚質内訳: {breakdown}")
        favored, tag = ["逃げ", "先行"], "逃げ・先行"
    else:
        st.info(f"⚖️ **ミドルペース想定**（ハイペース確率 {p['high_prob']:.1f}%）\n\n"
                "極端な前傾・後傾にはなりにくく、脚質より馬の能力が結果に出やすい展開です。\n\n"
                f"脚質内訳: {breakdown}")
        favored, tag = None, None

    if favored:
        picks = out[out["想定脚質"].isin(favored)].head(3)
        if len(picks):
            names = "、".join(f"{int(r['馬番'])}番 {r['馬名']}（{r['AI予想確率(%)']:.1f}%）"
                              for _, r in picks.iterrows())
            st.caption(f"📌 展開が向きそうな{tag}で確率上位: {names}")


# ------------------------------------------------------------------
# 特徴量の組み立て
# ------------------------------------------------------------------
DEFAULT_FEATURES = [
    "Venue", "Distance", "Track Condition", "Weather", "Bracket", "Horse Number",
    "Sex", "Horse Age", "Burden Weight", "Jockey", "Trainer", "Sire", "Tactics",
    "Total Prize", "Last Placing", "Last Time",
]


def build_X(df, feats):
    lookup = {norm(c): c for c in df.columns}
    X = pd.DataFrame(index=df.index)
    missing = []
    for f in feats:
        src = lookup.get(norm(f))
        if src is None:
            X[f] = 0.0
            missing.append(f)
        else:
            X[f] = df[src].values
    return X, missing


def get_pandas_categorical(model):
    """LightGBM が学習時に覚えた category 列のカテゴリ一覧（無ければ None）。"""
    for obj in (model, getattr(model, "booster_", None), getattr(model, "_Booster", None)):
        if obj is None:
            continue
        pc = getattr(obj, "pandas_categorical", None)
        if pc:
            return pc
    return None


def exact_categorical_columns(model, columns, n_expected):
    """モデル内部(feature_infos)から、学習時に category 型だった列名を学習時の列順で返す。
    特定できない／個数が合わない場合は None。"""
    for obj in (model, getattr(model, "booster_", None), getattr(model, "_Booster", None)):
        if obj is None or not hasattr(obj, "model_to_string"):
            continue
        try:
            text = obj.model_to_string()
            names = list(obj.feature_name())
            m = re.search(r"^feature_infos=(.*)$", text, re.M)
            if not m:
                continue
            infos = m.group(1).strip().split(" ")
            cols = [names[i] for i, x in enumerate(infos)
                    if i < len(names) and x != "none" and not x.startswith("[")]
            by_norm = {norm(c): c for c in columns}
            cols = [by_norm[norm(c)] for c in cols if norm(c) in by_norm]
            if len(cols) == n_expected:
                return cols
        except Exception:
            continue
    return None


def _find_encoder(meta, col):
    for k, enc in (meta.get("encoders") or {}).items():
        if norm(k) == norm(col) and hasattr(enc, "classes_"):
            return enc
    return None


def encode_X(X: pd.DataFrame, model=None, meta=None) -> pd.DataFrame:
    """モデルに渡す形へ変換。
    1) LightGBM が category 型で学習していた場合 → 同じカテゴリ定義の category 型にする
    2) LabelEncoder が pkl に入っている場合 → それで数値化（未知は -1）
    3) どちらも無ければ crc32 ハッシュで数値化
    """
    meta = meta or {}
    X = X.copy()
    str_cols = [c for c in X.columns if norm(c) in {norm(s) for s in STR_COLS}]
    pcat = get_pandas_categorical(model) if model is not None else None

    cat_map = {}  # 列名 -> カテゴリ一覧
    if pcat:
        exact = exact_categorical_columns(model, list(X.columns), len(pcat))
        if exact:
            for col, cats in zip(exact, pcat):
                cat_map[col] = list(cats)
    if pcat and not cat_map:
        cand = list(str_cols)
        if len(cand) < len(pcat):
            raise ValueError(
                f"モデルは category 列を {len(pcat)} 個使っていますが、"
                f"入力側の文字列列は {len(cand)} 個しかありません。")
        if len(cand) > len(pcat):
            # カテゴリ一覧との一致度が高い列を選ぶ（元の列順は保つ）
            def score(col, cats):
                cs = set(map(str, cats))
                return sum(str(v) in cs for v in X[col])
            chosen, used = [], set()
            for cats in pcat:
                best = max((c for c in cand if c not in used), key=lambda c: score(c, cats))
                used.add(best)
                chosen.append(best)
            cand = [c for c in X.columns if c in used]
        for col, cats in zip(cand, pcat):
            cat_map[col] = list(cats)

    for col in X.columns:
        if col in cat_map:
            cats = cat_map[col]
            as_str = all(isinstance(c, str) for c in cats)
            vals = [str(v) if as_str else v for v in X[col]]
            X[col] = pd.Categorical(vals, categories=cats)
        elif col in str_cols:
            enc = _find_encoder(meta, col)
            if enc is not None:
                table = {str(c): i for i, c in enumerate(enc.classes_)}
                X[col] = pd.Series([float(table.get(str(v), -1)) for v in X[col]],
                                   index=X.index, dtype="float64")
            else:
                X[col] = pd.Series(
                    [float(zlib.crc32(str(v).encode("utf-8")) % 100) for v in X[col]],
                    index=X.index, dtype="float64")
        else:
            X[col] = pd.Series(
                np.nan_to_num(to_float_array(X[col]), nan=0.0),
                index=X.index, dtype="float64",
            )
    return X


def predict_top3(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if p.ndim == 2 else p
    return np.clip(np.asarray(model.predict(X), dtype=float), 0, 1)


def check_gsheets_secrets():
    """Secrets の設定ミスを、原因が分かる日本語メッセージで知らせる。"""
    try:
        sec = st.secrets["connections"]["gsheets"]
    except Exception:
        raise RuntimeError(
            "Secrets に [connections.gsheets] がありません。"
            "Streamlit Cloud の Settings > Secrets に設定してください。")
    if str(sec.get("type", "")) != "service_account":
        raise RuntimeError(
            'Secrets に type = "service_account" がありません'
            "（無いと「公開シート」扱いになり書き込めません）。")
    need = ["spreadsheet", "project_id", "private_key_id", "private_key",
            "client_email", "client_id", "token_uri"]
    missing = [k for k in need if not str(sec.get(k, "")).strip()]
    if missing:
        raise RuntimeError("Secrets に次の項目がありません: " + ", ".join(missing))
    if "BEGIN PRIVATE KEY" not in str(sec["private_key"]):
        raise RuntimeError("private_key の形式が正しくありません"
                           "（-----BEGIN PRIVATE KEY----- で始まる値をそのまま貼ってください）。")


LOG_COLS = [
    "Execution Date", "Race Day", "Race Name", "Venue", "Distance",
    "Track Condition", "Weather", "Bracket", "Horse Number", "Horse Name",
    "Sex", "Horse Age", "Burden Weight", "Jockey", "Trainer", "Tactics",
    "AI_Probability", "AI_Rank", "Placing",
]


def build_log(df, proba, now, race_day, race_name, venue, distance, track, weather):
    """前処理済み出馬表 df と AI予想を、1頭1行・全19列のログに組み立てる（再学習用）。"""
    prob = np.round(np.asarray(proba, dtype=float) * 100, 1)
    # 画面の順位表と同じ並び（確率降順・同率は元の行順）で順位を付ける
    order = np.argsort(-prob, kind="stable")
    rank = np.empty(len(prob), dtype=int)
    rank[order] = np.arange(1, len(prob) + 1)

    n = len(df)
    log = pd.DataFrame({
        "Execution Date": [now] * n,
        "Race Day": [str(race_day)] * n,
        "Race Name": [race_name or ""] * n,
        "Venue": [venue] * n,
        "Distance": [int(distance)] * n,
        "Track Condition": [track] * n,
        "Weather": [weather] * n,
        "Bracket": df["Bracket"].astype(int).values,
        "Horse Number": df["Horse Number"].astype(int).values,
        "Horse Name": df["Horse Name"].values,
        "Sex": df["Sex"].values,
        "Horse Age": df["Horse Age"].astype(int).values,
        "Burden Weight": df["Burden Weight"].astype(float).values,
        "Jockey": df["Jockey"].values,
        "Trainer": df["Trainer"].values,
        # 脚質は 1〜4。読み取れなかった馬は空欄にする
        "Tactics": pd.array([int(x) if pd.notna(x) else pd.NA for x in df["TacticsCode"]], dtype="Int64"),
        "AI_Probability": prob,
        "AI_Rank": rank,
        "Placing": [None] * n,  # レース後に手動で着順を追記
    })
    log = log.sort_values("Horse Number", kind="stable").reset_index(drop=True)
    return log[LOG_COLS]


def append_to_sheet(new_rows: pd.DataFrame):
    check_gsheets_secrets()
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        existing = conn.read(worksheet="predictions", ttl=0)
        existing = existing.dropna(how="all").dropna(axis=1, how="all") if existing is not None else pd.DataFrame()
    except Exception:
        existing = pd.DataFrame()

    if len(existing):
        existing = _clean_columns(existing)  # 手入力の過去データ（結果のみ行）の列名揺れを吸収
    combined = pd.concat([existing, new_rows], ignore_index=True)
    # 日付は YYYY-MM-DD に統一（シート側で日付型になっていても崩さない）
    combined["Race Day"] = [_day_str(v) for v in combined["Race Day"]]
    # 列順を仕様どおりに固定（旧形式の列があれば右側に残す）
    for c in LOG_COLS:
        if c not in combined.columns:
            combined[c] = None
    extra = [c for c in combined.columns if c not in LOG_COLS]
    combined = combined[LOG_COLS + extra]
    # 空欄（None/NaN）は空文字で書き出す（Placing は空欄のまま）
    combined = combined.astype(object).where(combined.notna(), "")
    conn.update(worksheet="predictions", data=combined)


# ------------------------------------------------------------------
# 過去ログ・精度検証
# ------------------------------------------------------------------
def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名の空白除去・文字列化・重複列の除去（シート手入力の揺れ対策）。"""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df.loc[:, ~pd.Index(df.columns).duplicated()]


def load_history() -> pd.DataFrame:
    check_gsheets_secrets()
    conn = st.connection("gsheets", type=GSheetsConnection)
    raw = conn.read(worksheet="predictions", ttl=0)
    if raw is None:
        return pd.DataFrame()
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    return _clean_columns(raw)


def _parse_day(v):
    """日付の表記揺れ（2021-07-03 / 2021/7/3 / 2021年7月3日 / Excelシリアル値）を Timestamp に。"""
    if _is_missing(v):
        return pd.NaT
    if isinstance(v, bool):
        return pd.NaT
    if isinstance(v, (int, float, np.integer, np.floating)):
        if 30000 < float(v) < 80000:  # Excel のシリアル値
            return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(v))
        return pd.NaT
    if not isinstance(v, str):
        try:
            return pd.Timestamp(v)
        except Exception:
            return pd.NaT
    s = v.strip().replace("日", "")
    s = re.sub(r"[年月/.]", "-", s).strip("- ")
    if not s:
        return pd.NaT
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT


def _day_str(v) -> str:
    """日付は YYYY-MM-DD の文字列に統一。解釈できなければ元の文字列のまま。"""
    if _is_missing(v):
        return ""
    ts = _parse_day(v)
    return ts.strftime("%Y-%m-%d") if pd.notna(ts) else str(v)


def to_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """st.dataframe 用: 数値以外の列は全て文字列化（数値と文字の混在によるArrowエラー防止）。"""
    out = df.copy()
    out.columns = [str(c) for c in out.columns]
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            continue
        out[c] = pd.Series(["" if _is_missing(v) else str(v) for v in s],
                           index=out.index, dtype="object")
    return out


def prepare_history(raw: pd.DataFrame) -> pd.DataFrame:
    """予想なし（結果のみ）の行や、空欄・文字・None の混在があっても落ちないよう集計用の列を作る。"""
    h = _clean_columns(raw)
    for c in LOG_COLS:
        if c not in h.columns:
            h[c] = np.nan

    def num(col):
        return pd.Series(to_float_array(h[col]), index=h.index, dtype="float64")

    h["_placing"] = num("Placing")
    h.loc[h["_placing"] < 1, "_placing"] = np.nan   # 0 や負数は結果なし扱い
    h["_rank"] = num("AI_Rank")
    h.loc[h["_rank"] < 1, "_rank"] = np.nan         # 予想なし（空欄）は NaN のまま
    h["_prob"] = num("AI_Probability")
    h["_hn"] = num("Horse Number")

    days = h["Race Day"]
    cache = {u: _parse_day(u) for u in days.dropna().unique()}
    h["_day"] = pd.to_datetime(days.map(cache), errors="coerce")
    # 並べ替え用: 型が混在していても比較で落ちないよう文字列化
    h["_exec"] = ["" if _is_missing(v) else str(v) for v in h["Execution Date"]]
    return h


def _count(d: pd.DataFrame) -> dict:
    return {
        "n": int(len(d)),
        "win": int((d["_placing"] == 1).sum()),
        "ren": int((d["_placing"] <= 2).sum()),
        "fuku": int((d["_placing"] <= 3).sum()),
    }


def compute_stats(h: pd.DataFrame):
    """「AI_Rank が存在」かつ「Placing が入力済み」の行だけを集計対象にする。無ければ None。"""
    done = h[h["_placing"].notna() & h["_rank"].notna()]
    if done.empty:
        return None
    keys = done[["Execution Date", "Race Day", "Race Name"]].astype(str).agg("|".join, axis=1)
    p = done["_prob"]
    buckets = [
        ("70%以上", done[p >= 70]),
        ("50〜69%", done[(p >= 50) & (p < 70)]),
        ("50%未満", done[p < 50]),
    ]
    return {
        "n_races": int(keys.nunique()),
        "n_horses": int(len(done)),
        "honmei": _count(done[done["_rank"] == 1]),
        "top3": _count(done[done["_rank"] <= 3]),
        "buckets": [(name, _count(d)) for name, d in buckets],
    }


def _pct(num: int, den: int) -> str:
    return f"{num / den * 100:.1f}%" if den else "—"


def _pct_val(num: int, den: int):
    return round(num / den * 100, 1) if den else None


def render_history_tab():
    if st.button("🔄 最新データを読み込む", key="hist_reload"):
        st.session_state.pop("history_df", None)
        st.session_state.pop("history_err", None)

    # 絞り込みのたびにAPIを叩かないよう、読み込み結果は session_state に保持する
    if "history_df" not in st.session_state:
        try:
            with st.spinner("スプレッドシートを読み込み中…"):
                st.session_state["history_df"] = load_history()
            st.session_state.pop("history_err", None)
        except Exception as e:
            st.session_state["history_df"] = None
            st.session_state["history_err"] = str(e)

    raw = st.session_state.get("history_df")
    if raw is None:
        st.error(f"過去ログの読み込みに失敗しました: {st.session_state.get('history_err', '')}")
        return
    if raw.empty:
        st.info("まだ保存された予想がありません。予想を実行するとここに蓄積されます。")
        return

    try:
        h = prepare_history(raw)
    except Exception as e:
        st.error(f"過去ログの整形中にエラーが発生しました: {e}")
        st.dataframe(to_display_df(raw), hide_index=True, use_container_width=True, height=420)
        return

    # ---- 検索・フィルタ ----
    st.subheader("過去の予想を探す")
    f1, f2 = st.columns(2)
    with f1:
        q = st.text_input("レース名検索", key="hist_q", placeholder="例：菊花賞")
    with f2:
        venues = sorted(set(h["Venue"].dropna().astype(str)) - {""})
        sel_v = st.multiselect("競馬場フィルタ（未選択＝すべて）", venues, key="hist_v")

    mask = pd.Series(True, index=h.index)
    if q.strip():
        mask &= h["Race Name"].astype(str).str.contains(q.strip(), case=False, na=False, regex=False)
    if sel_v:
        mask &= h["Venue"].astype(str).isin(sel_v)

    valid_days = h["_day"].dropna()
    if not valid_days.empty:
        dmin, dmax = valid_days.min().date(), valid_days.max().date()
        if dmin < dmax:
            rng = st.date_input("日付フィルタ（レース開催日）", value=(dmin, dmax),
                                min_value=dmin, max_value=dmax, key="hist_d")
            if isinstance(rng, (tuple, list)) and len(rng) > 0:
                start = rng[0]
                end = rng[1] if len(rng) > 1 else rng[0]
                if (start, end) != (dmin, dmax):
                    d = h["_day"].dt.date
                    mask &= d.notna() & (d >= start) & (d <= end)
        else:
            st.caption(f"開催日: {dmin}（1日分のみ）")

    only_done = st.checkbox("結果（Placing）入力済みのみ表示", key="hist_done")
    if only_done:
        mask &= h["_placing"].notna()

    view = h[mask].copy()
    view = view.sort_values(["_day", "_exec", "_hn"],
                            ascending=[False, False, True], na_position="last")

    # 予想なし（AI_Probability 等が空欄）の行も、レース名・日付・馬名・着順などをそのまま閲覧可能
    raw_cols = [str(c) for c in raw.columns]
    show_cols = [c for c in LOG_COLS if c in raw_cols] + [c for c in raw_cols if c not in LOG_COLS]
    shown = view[show_cols].copy()
    if "Race Day" in shown.columns:
        shown["Race Day"] = [
            ts.strftime("%Y-%m-%d") if pd.notna(ts) else ("" if _is_missing(orig) else str(orig))
            for ts, orig in zip(view["_day"], shown["Race Day"])
        ]

    n_pred = int(view["_rank"].notna().sum())
    st.caption(f"該当 {len(view)} 頭 / 全 {len(h)} 頭（うちAI予想あり {n_pred} 頭、結果のみ {len(view) - n_pred} 頭）")
    st.dataframe(to_display_df(shown), hide_index=True, use_container_width=True, height=420)

    # ---- 成績集計（AI_Rank あり かつ Placing 入力済み） ----
    st.subheader("AIの的中率・成績")
    st.caption("※上の絞り込み条件を反映し、AI予想（AI_Rank）があり、かつ Placing に数値が入っている行だけを集計します。"
               "AI_Probability は「3着以内に入る確率」です。")
    n_results_only = int((view["_placing"].notna() & view["_rank"].isna()).sum())
    if n_results_only:
        st.caption(f"予想データのない結果のみの {n_results_only} 頭は、集計から除外しています。")

    stats = compute_stats(view)
    if stats is None:
        st.info("スプレッドシートのPlacing列に結果を入力すると、ここに的中率が表示されます")
        return

    m1, m2 = st.columns(2)
    m1.metric("検証レース数", f"{stats['n_races']}")
    m2.metric("検証頭数", f"{stats['n_horses']}")

    st.markdown("**AI本命（AI_Rank = 1）の成績**")
    hon = stats["honmei"]
    a, b, c = st.columns(3)
    a.metric("勝率（1着）", _pct(hon["win"], hon["n"]))
    b.metric("連対率（2着以内）", _pct(hon["ren"], hon["n"]))
    c.metric("複勝率（3着以内）", _pct(hon["fuku"], hon["n"]))
    st.caption(f"本命 {hon['n']}頭中: 1着 {hon['win']} / 2着以内 {hon['ren']} / 3着以内 {hon['fuku']}")

    st.markdown("**AI予想上位（AI_Rank ≦ 3）**")
    t3 = stats["top3"]
    st.metric("複勝率（3着以内）", _pct(t3["fuku"], t3["n"]))
    st.caption(f"上位3頭 計{t3['n']}頭中、3着以内 {t3['fuku']}頭")

    st.markdown("**予想確率別の的中率**")
    rows = [{
        "予想確率": name, "頭数": d["n"],
        "勝率(%)": _pct_val(d["win"], d["n"]),
        "複勝率(%)": _pct_val(d["fuku"], d["n"]),
    } for name, d in stats["buckets"]]
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, use_container_width=True,
        column_config={
            "勝率(%)": st.column_config.ProgressColumn("勝率(%)", format="%.1f%%", min_value=0, max_value=100),
            "複勝率(%)": st.column_config.ProgressColumn("複勝率(%)", format="%.1f%%", min_value=0, max_value=100),
        },
    )


# ------------------------------------------------------------------
# 画面メイン描画処理
# ------------------------------------------------------------------
tab_run, tab_hist = st.tabs(["🔮 AI予想を実行", "📚 過去ログ＆的中率確認"])

with tab_hist:
    render_history_tab()

with tab_run:
    try:
        model, meta = load_model()
    except Exception as e:
        st.error(f"model.pkl の読み込みに失敗しました: {e}")
        st.stop()

    st.subheader("① レース情報")
    c1, c2 = st.columns(2)
    with c1:
        venue = st.selectbox("競馬場（Venue）", VENUES, index=VENUES.index("東京"))
        track = st.selectbox("馬場状態（Track Condition）", TRACKS)
    with c2:
        distance = st.selectbox("距離（Distance）", DISTANCES, index=DISTANCES.index(1600))
        weather = st.selectbox("天気（Weather）", WEATHERS)
    race_name = st.text_input("レース名（任意・記録用）", placeholder="例：菊花賞")
    race_day = st.date_input("レース開催日（記録用）", value=datetime.now(JST).date())

    st.subheader("② 出馬表を貼り付け")
    csv_text = st.text_area(
        "出馬表CSV",
        height=320,
        label_visibility="collapsed",
        placeholder="ここに整形したCSVデータを貼り付けてください",
    )

    run = st.button("🔮 AI予想を実行する", type="primary", use_container_width=True)

    if run:
        st.session_state.pop("saved", None)
        st.session_state.pop("save_error", None)
        if not csv_text.strip():
            st.warning("出馬表CSVを貼り付けてください。")
        else:
            try:
                raw = parse_csv(csv_text)
                df = preprocess(raw, venue, distance, track, weather)
                pace = diagnose_pace(df)

                feats = get_feature_names(model, meta) or DEFAULT_FEATURES
                X, missing = build_X(df, feats)
                X = encode_X(X, model, meta)
                if X.shape[0] == 0 or X.shape[1] == 0:
                    raise ValueError(f"モデルへの入力が空です（行数={X.shape[0]}, 列数={X.shape[1]}）。")

                proba = predict_top3(model, X[feats])

                now = datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")
                out = pd.DataFrame({
                    "馬番": df["Horse Number"].astype(int),
                    "馬名": df["Horse Name"],
                    "種牡馬": df["Sire"],
                    "想定脚質": df["脚質ラベル"],
                    "AI予想確率(%)": np.round(np.asarray(proba, dtype=float) * 100, 1),
                })
                out = out.sort_values("AI予想確率(%)", ascending=False, kind="stable").reset_index(drop=True)
                out.insert(0, "順位", np.arange(1, len(out) + 1))

                st.session_state["result"] = out
                st.session_state["pace"] = pace

                # スプレッドシートに蓄積するログ（全19列・1頭1行）を作成
                log = build_log(df, proba, now, race_day, race_name, venue, distance, track, weather)

                try:
                    append_to_sheet(log)
                    st.session_state["saved"] = True
                    st.session_state.pop("history_df", None)  # 次回表示時に最新ログを再読込
                except Exception as e:
                    st.session_state["saved"] = False
                    st.session_state["save_error"] = str(e)

            except Exception as e:
                st.session_state.pop("result", None)
                st.error(f"予想中にエラーが発生しました: {e}")

    # 🔮 画面への最終結果出力エリア
    if "result" in st.session_state:
        out = st.session_state["result"]
        render_pace(st.session_state.get("pace", {"ok": False}), out)

        st.subheader(f"④ AI予想結果（全{len(out)}頭・3着以内確率順）")
        st.dataframe(
            out,
            hide_index=True,
            use_container_width=True,
            height=int(38 * (len(out) + 1)),
            column_config={
                "順位": st.column_config.NumberColumn("順位", width="small"),
                "馬番": st.column_config.NumberColumn("馬番", width="small"),
                "AI予想確率(%)": st.column_config.ProgressColumn(
                    "AI予想確率(%)", format="%.1f%%", min_value=0, max_value=100
                ),
            },
        )
        if st.session_state.get("saved"):
            st.success("データ蓄積完了")
        elif st.session_state.get("saved") is False:
            st.warning(f"スプレッドシートへの保存に失敗しました: {st.session_state.get('save_error', '')}")
