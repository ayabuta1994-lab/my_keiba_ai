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


def encode_X(X: pd.DataFrame) -> pd.DataFrame:
    """全列を float に統一（カテゴリ列は crc32 ハッシュで数値化）。"""
    X = X.copy()
    for col in X.columns:
        if col in CAT_COLS:
            X[col] = pd.Series(
                [float(zlib.crc32(str(v).encode("utf-8")) % 100) for v in X[col]],
                index=X.index, dtype="float64",
            )
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


def append_to_sheet(new_rows: pd.DataFrame):
    conn = st.connection("gsheets", type=GSheetsConnection)
    try:
        existing = conn.read(worksheet="predictions", ttl=0)
        existing = existing.dropna(how="all").dropna(axis=1, how="all") if existing is not None else pd.DataFrame()
    except Exception:
        existing = pd.DataFrame()
    combined = pd.concat([existing, new_rows], ignore_index=True)
    conn.update(worksheet="predictions", data=combined)


# ------------------------------------------------------------------
# 画面メイン描画処理
# ------------------------------------------------------------------
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
            X = encode_X(X)
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

            # スプレッドシートに蓄積するためのログを作成
            log = out.copy()
            log.insert(0, "実行日時", now)
            log.insert(1, "レース名", race_name)
            log.insert(2, "競馬場", venue)
            log.insert(3, "距離", distance)
            log.insert(4, "馬場状態", track)
            log.insert(5, "天気", weather)
            log.insert(6, "想定ペース", pace.get("pace", "不明"))

            try:
                append_to_sheet(log)
                st.session_state["saved"] = True
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
