# -*- coding: utf-8 -*-
"""
🏇 自分専用 競馬予想AI（オールインワン版）
- 出走表タブ = 一時バッファ（予想保存後に自動クリア / 予想訂正で再読み込み）
- race_results   = AI予想結果を全件保存（結果登録時に実績を追記）
- predictions    = 学習用データ（レース結果 + AI予想の勝率・予想順位のみ）
- race_management= Race IDをユニークキーとした一元管理（未予想 → 予想済み → 結果登録済み）
- スプレッドシートはボタン押下時のみ読み込む（遅延ロード）
"""
import base64
import io
import pickle
import re
import traceback
import zlib
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials

st.set_page_config(page_title="競馬予想AI オールインワン", page_icon="🐴", layout="wide")

MODEL_PATH = Path(__file__).parent / "model.pkl"
JST = timezone(timedelta(hours=9))

def _st_version():
    m = re.match(r"(\d+)\.(\d+)", str(getattr(st, "__version__", "0.0")))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


# Streamlit 1.50+ は width="stretch"、それ未満は use_container_width=True を使う
STRETCH = {"width": "stretch"} if _st_version() >= (1, 50) else {"use_container_width": True}

# ============================================================
# 定数
# ============================================================
# 統一する21列の固定ヘッダー定義（変更しない）
COLUMNS_21 = [
    "Execution Date", "Race Day", "Race Name", "Venue", "Distance",
    "Track Condition", "Weather", "Bracket", "Horse Number", "Horse Name",
    "Sex", "Horse Age", "Burden Weight", "Jockey", "Trainer",
    "Tactics", "Last 3F", "AI_Probability", "AI_Rank", "Placing", "Track_Type",
]

NUM_DEFAULTS = {
    "Bracket": 0, "Horse Number": 0, "Horse Age": 4, "Burden Weight": 55,
    "Last 3F": "mean:36.5", "Distance": 1800,
}
STR_COLS = ["Horse Name", "Sex", "Jockey", "Trainer", "Venue", "Track Condition", "Weather"]
COLUMNS_HORSES = ["Horse Name", "Sire", "Dam's Sire"]

# シート名
ENTRY_SHEET = "出走表"
PREDICTIONS_SHEET = "predictions"
RESULTS_SHEET = "race_results"
RACE_MANAGEMENT_SHEET = "race_management"
HORSES_SHEET = "horses_master"
# 学習履歴・タブ一覧から除外する管理用シート
SYSTEM_SHEETS = (HORSES_SHEET, RACE_MANAGEMENT_SHEET, RESULTS_SHEET, ENTRY_SHEET)

# race_results: 先頭11列は従来の「実績」列（互換維持）、以降がAI予想の全項目
RESULT_COLUMNS_LEGACY = [
    "Race ID", "Race Day", "Venue", "Race Name", "Distance", "Horse Number",
    "Horse Name", "Tactics", "Last 3F", "Placing", "Updated At",
]
RESULT_COLUMNS = RESULT_COLUMNS_LEGACY + [
    "Execution Date", "Track Condition", "Weather", "Bracket", "Sex", "Horse Age",
    "Burden Weight", "Jockey", "Trainer", "Track_Type",
    "Input Tactics", "Input Last 3F",
    "AI_Probability", "AI_Rank", "Top3 Probability",
    "Pace", "High Pace Prob", "Front Ratio", "Predicted At",
]

# race_management: 先頭8列は従来列（互換維持）
RACE_MANAGEMENT_COLUMNS_LEGACY = [
    "Race ID", "Race Day", "Venue", "Race Name", "Distance", "Status", "Horse Count", "Registered At",
]
RACE_MANAGEMENT_COLUMNS = RACE_MANAGEMENT_COLUMNS_LEGACY + ["Updated At"]

STATUS_UNPREDICTED = "未予想"
STATUS_PREDICTED = "予想済み"
STATUS_RESULT = "結果登録済み"
STATUS_ORDER = {STATUS_UNPREDICTED: 0, STATUS_PREDICTED: 1, STATUS_RESULT: 2}
STATUS_LEGACY_MAP = {"予想登録済み": STATUS_PREDICTED, "": STATUS_UNPREDICTED}

VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉", "その他"]
TRACKS = ["良", "稍重", "重", "不良"]
WEATHERS = ["晴", "曇", "小雨", "雨", "雪"]
TRACK_TYPES = {"ダート": "1", "芝": "0", "障害": "2"}
TACTICS_LABEL = {1: "逃げ", 2: "先行", 3: "差し", 4: "追込"}

# 出走表エディタで編集する列
EDITOR_COLUMNS = [
    "Horse Number", "Horse Name", "Bracket", "Sex", "Horse Age",
    "Burden Weight", "Jockey", "Trainer", "Tactics", "Last 3F",
]


# ============================================================
# 汎用ユーティリティ
# ============================================================
def norm(s) -> str:
    return re.sub(r"[\s_\-]", "", str(s)).lower()


def _norm_horse(v):
    return re.sub(r"[\s\u200b-\u200f\u2060\ufeff]", "", str(v or ""))


def now_str() -> str:
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M:%S")


def _norm_status(v) -> str:
    s = str(v or "").strip()
    return STATUS_LEGACY_MAP.get(s, s)


def _norm_date_key(v) -> str:
    """日付を YYYYMMDD に正規化（解釈できなければ記号除去のみ）。"""
    s = str(v or "").strip()
    d = pd.to_datetime(s, errors="coerce")
    if pd.notna(d):
        return d.strftime("%Y%m%d")
    return re.sub(r"[^\w]+", "", s)[:24]


def _norm_distance_key(v) -> str:
    s = str(v or "").replace(",", "").strip()
    m = re.search(r"\d+", s)
    return m.group(0) if m else s


def _race_id(race_day, venue, race_name, distance="") -> str:
    """Race ID（ユニークキー）。21列データの外で管理する。"""
    parts = [
        _norm_date_key(race_day),
        re.sub(r"[^\w一-龯ぁ-んァ-ン]+", "", str(venue).strip())[:24],
        re.sub(r"[^\w一-龯ぁ-んァ-ン]+", "", str(race_name).strip())[:24],
        _norm_distance_key(distance),
    ]
    if not any(parts):
        return ""
    return "_".join(parts)


def _race_id_from_row(row) -> str:
    return _race_id(row.get("Race Day", ""), row.get("Venue", ""), row.get("Race Name", ""), row.get("Distance", ""))


def _race_label(row) -> str:
    return f"{row.get('Race Day', '')}｜{row.get('Venue', '')}｜{row.get('Race Name', '')}（{row.get('Distance', '')}m）"


def _col_letter(n: int) -> str:
    """1始まりの列番号 → A1表記の列記号。"""
    return gspread.utils.rowcol_to_a1(1, n).rstrip("0123456789")


def decode_bytes(b: bytes) -> str:
    for enc in ("utf-8-sig", "cp932", "utf-16"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="replace")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Excelで文字化けしない UTF-8 with BOM。"""
    return df.to_csv(index=False).encode("utf-8-sig")


# ============================================================
# Google スプレッドシート接続・読み込み（遅延ロード）
# ============================================================
@st.cache_resource
def get_gspread_client():
    creds_dict = None
    for key in ["connections.gsheets", "gcp_service_account", "gsheets"]:
        if key in st.secrets:
            creds_dict = dict(st.secrets[key])
            break

    if creds_dict is None:
        if hasattr(st.secrets, "connections") and "gsheets" in st.secrets.connections:
            creds_dict = dict(st.secrets.connections.gsheets)
        else:
            raise KeyError("Streamlit Secrets にサービスアカウント情報が見つかりません。")

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    gc = gspread.authorize(creds)

    spreadsheet_ref = creds_dict.get("spreadsheet") or st.secrets.get("spreadsheet_id")
    if not spreadsheet_ref:
        raise ValueError("Secrets に spreadsheet のURLまたはIDが設定されていません。")

    if "spreadsheets/d/" in str(spreadsheet_ref):
        return gc.open_by_url(spreadsheet_ref)
    return gc.open_by_key(spreadsheet_ref)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_sheet_values(spreadsheet_id, worksheet_title):
    """429対策の短時間キャッシュ。読み込みボタン押下時は clear_sheet_cache() で破棄する。"""
    sh = get_gspread_client()
    return sh.worksheet(worksheet_title).get_all_values()


@st.cache_data(ttl=30, show_spinner=False)
def _cached_worksheet_titles(spreadsheet_id):
    sh = get_gspread_client()
    return [ws.title for ws in sh.worksheets()]


def clear_sheet_cache():
    _cached_sheet_values.clear()
    _cached_worksheet_titles.clear()


def values_to_dataframe(values) -> pd.DataFrame:
    """get_all_values() の結果を DataFrame に変換する（全て文字列）。"""
    if not values or len(values) < 2:
        return pd.DataFrame()
    headers = [str(v).strip() for v in values[0]]
    while headers and not headers[-1]:
        headers.pop()
    if not headers:
        return pd.DataFrame()
    rows = []
    for row in values[1:]:
        row = list(row[: len(headers)])
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))
        if any(str(x).strip() for x in row):
            rows.append(row)
    return pd.DataFrame(rows, columns=headers)


def _read_worksheet_dataframe(ws) -> pd.DataFrame:
    return values_to_dataframe(_cached_sheet_values(ws.spreadsheet.id, ws.title))


def read_sheet_df(spreadsheet, title: str, fresh: bool = False) -> pd.DataFrame:
    """シートを DataFrame で取得。シートが無い場合は空 DataFrame。"""
    if fresh:
        clear_sheet_cache()
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        return pd.DataFrame()
    return _read_worksheet_dataframe(ws)


def _ensure_sheet(spreadsheet, title, columns, legacy_columns, rows=2000):
    """
    ヘッダー付きシートを用意する。
    - 無ければ作成
    - 従来ヘッダー（先頭の一部列）と完全一致する場合のみ、新列を右側に追加して拡張
    - それ以外でヘッダーが異なる場合は、既存データ保護のため例外
    """
    try:
        ws = spreadsheet.worksheet(title)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=title, rows=rows, cols=len(columns))
        ws.update(range_name="A1", values=[columns], value_input_option="USER_ENTERED")
        clear_sheet_cache()
        return ws

    header = [str(v).strip() for v in ws.row_values(1)]
    while header and not header[-1]:
        header.pop()
    if not header:
        ws.update(range_name="A1", values=[columns], value_input_option="USER_ENTERED")
        clear_sheet_cache()
    elif header == columns:
        pass
    elif header == legacy_columns or (len(header) < len(columns) and columns[: len(header)] == header and len(header) >= len(legacy_columns)):
        if ws.col_count < len(columns):
            ws.add_cols(len(columns) - ws.col_count)
        ws.update(range_name="A1", values=[columns], value_input_option="USER_ENTERED")
        clear_sheet_cache()
    else:
        raise ValueError(f"「{title}」タブのヘッダーが想定と異なります。既存データを保護するため自動変更しません。")
    return ws


def ensure_results_sheet(spreadsheet):
    return _ensure_sheet(spreadsheet, RESULTS_SHEET, RESULT_COLUMNS, RESULT_COLUMNS_LEGACY, rows=5000)


def ensure_management_sheet(spreadsheet):
    return _ensure_sheet(spreadsheet, RACE_MANAGEMENT_SHEET, RACE_MANAGEMENT_COLUMNS, RACE_MANAGEMENT_COLUMNS_LEGACY, rows=1000)


def ensure_entry_sheet(spreadsheet):
    """出走表（バッファ）シート。ヘッダーは21列固定。"""
    try:
        ws = spreadsheet.worksheet(ENTRY_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=ENTRY_SHEET, rows=200, cols=len(COLUMNS_21))
        ws.update(range_name="A1", values=[COLUMNS_21], value_input_option="USER_ENTERED")
        clear_sheet_cache()
        return ws
    if not any(str(v).strip() for v in ws.row_values(1)):
        ws.update(range_name="A1", values=[COLUMNS_21], value_input_option="USER_ENTERED")
        clear_sheet_cache()
    return ws


def init_empty_headers(spreadsheet):
    """
    各タブの1行目が空欄の場合のみ、必要な項目名を入力する。
    1行目に既に何か入力されているタブは一切変更しない。タブが無い場合は新規作成する。
    """
    targets = [
        (ENTRY_SHEET, COLUMNS_21), (PREDICTIONS_SHEET, COLUMNS_21), (RESULTS_SHEET, RESULT_COLUMNS),
        (RACE_MANAGEMENT_SHEET, RACE_MANAGEMENT_COLUMNS), (HORSES_SHEET, COLUMNS_HORSES),
    ]
    existing = {w.title for w in spreadsheet.worksheets()}
    report = []
    for title, cols in targets:
        if title not in existing:
            ws = spreadsheet.add_worksheet(title=title, rows=1000, cols=len(cols))
            ws.update(range_name="A1", values=[cols], value_input_option="USER_ENTERED")
            report.append((title, "タブが無かったため新規作成し、項目名を入力しました"))
            continue
        ws = spreadsheet.worksheet(title)
        header = [str(v).strip() for v in ws.row_values(1)]
        if any(header):
            report.append((title, "1行目に入力済みのため変更しませんでした"))
            continue
        if ws.col_count < len(cols):
            ws.add_cols(len(cols) - ws.col_count)
        ws.update(range_name="A1", values=[cols], value_input_option="USER_ENTERED")
        report.append((title, "1行目が空欄だったため項目名を入力しました"))
    clear_sheet_cache()
    return report


def clear_entry_buffer(spreadsheet):
    """出走表の2行目以降をクリアし、1行目（ヘッダー）のみ残す。"""
    ws = ensure_entry_sheet(spreadsheet)
    last_col = _col_letter(max(ws.col_count, len(COLUMNS_21)))
    if ws.row_count >= 2:
        ws.batch_clear([f"A2:{last_col}{ws.row_count}"])
    clear_sheet_cache()


def write_entry_buffer(spreadsheet, df21: pd.DataFrame, replace: bool = True):
    """出走表バッファへ書き込む（replace=True で既存データ行を置き換え）。"""
    ws = ensure_entry_sheet(spreadsheet)
    if replace:
        clear_entry_buffer(spreadsheet)
    rows = df21.fillna("").astype(str).values.tolist()
    if rows:
        ws.append_rows(rows, value_input_option="USER_ENTERED", table_range="A1")
    clear_sheet_cache()
    return len(rows)


def _delete_rows_by_numbers(ws, row_numbers):
    """行番号（1始まり）のリストを、下から連続ブロック単位で削除する。"""
    nums = sorted(set(int(n) for n in row_numbers), reverse=True)
    if not nums:
        return
    blocks, start, prev = [], nums[0], nums[0]
    for n in nums[1:]:
        if n == prev - 1:
            prev = n
        else:
            blocks.append((prev, start))
            start = prev = n
    blocks.append((prev, start))
    for lo, hi in blocks:  # 既に下側から並んでいるので行ズレしない
        ws.delete_rows(lo, hi)


# ============================================================
# race_management（Race IDをユニークキーとして管理）
# ============================================================
def _management_row(values, headers, row):
    d = {h: "" for h in RACE_MANAGEMENT_COLUMNS}
    for i, h in enumerate(headers):
        if h in d and i < len(row):
            d[h] = row[i]
    return d


def upsert_race_management(spreadsheet, race_id, meta, status, horse_count=None, force=False):
    """
    Race ID で行を探し、無ければ追加・有れば更新する（重複登録は発生しない）。
    ステータスは 未予想 → 予想済み → 結果登録済み の方向にのみ自動更新（force=True で手動変更）。
    書き込み前に必ずキャッシュを使わず最新値を読み直す。
    """
    if not race_id:
        raise ValueError("Race ID を生成できません（Race Day / Venue / Race Name を確認してください）。")
    ws = ensure_management_sheet(spreadsheet)
    values = ws.get_all_values()
    headers = [str(v).strip() for v in values[0]] if values else RACE_MANAGEMENT_COLUMNS
    idx = {h: i for i, h in enumerate(headers)}
    last_col = _col_letter(len(RACE_MANAGEMENT_COLUMNS))
    status = _norm_status(status)

    hit_rows = []
    for rn, row in enumerate(values[1:], start=2):
        if row and str(row[idx["Race ID"]]).strip() == race_id:
            hit_rows.append(rn)

    if hit_rows:
        rn = hit_rows[0]
        cur = _management_row(values, headers, values[rn - 1])
        cur_status = _norm_status(cur["Status"])
        new_status = status if (force or STATUS_ORDER.get(status, 0) >= STATUS_ORDER.get(cur_status, 0)) else cur_status
        cur.update({
            "Race Day": meta.get("Race Day", cur["Race Day"]) or cur["Race Day"],
            "Venue": meta.get("Venue", cur["Venue"]) or cur["Venue"],
            "Race Name": meta.get("Race Name", cur["Race Name"]) or cur["Race Name"],
            "Distance": meta.get("Distance", cur["Distance"]) or cur["Distance"],
            "Status": new_status,
            "Horse Count": str(horse_count) if horse_count is not None else cur["Horse Count"],
            "Updated At": now_str(),
        })
        ws.update(range_name=f"A{rn}:{last_col}{rn}", values=[[cur[c] for c in RACE_MANAGEMENT_COLUMNS]], value_input_option="USER_ENTERED")
        # 万一の重複行は削除して一意性を保つ
        if len(hit_rows) > 1:
            _delete_rows_by_numbers(ws, hit_rows[1:])
        clear_sheet_cache()
        return "updated", new_status

    row = {
        "Race ID": race_id, "Race Day": meta.get("Race Day", ""), "Venue": meta.get("Venue", ""),
        "Race Name": meta.get("Race Name", ""), "Distance": meta.get("Distance", ""),
        "Status": status, "Horse Count": str(horse_count) if horse_count is not None else "",
        "Registered At": now_str(), "Updated At": now_str(),
    }
    ws.append_row([row[c] for c in RACE_MANAGEMENT_COLUMNS], value_input_option="USER_ENTERED", table_range="A1")
    clear_sheet_cache()
    return "created", status


def load_race_management_df(spreadsheet, fresh=True) -> pd.DataFrame:
    ensure_management_sheet(spreadsheet)
    df = read_sheet_df(spreadsheet, RACE_MANAGEMENT_SHEET, fresh=fresh)
    if df.empty:
        return pd.DataFrame(columns=RACE_MANAGEMENT_COLUMNS)
    for c in RACE_MANAGEMENT_COLUMNS:
        if c not in df.columns:
            df[c] = ""
    df = df[RACE_MANAGEMENT_COLUMNS].fillna("").astype(str)
    df["Status"] = df["Status"].map(_norm_status)
    df = df[df["Race ID"].str.strip() != ""].drop_duplicates(subset=["Race ID"], keep="first")
    return df.reset_index(drop=True)


def backfill_management_from_predictions(spreadsheet):
    """既存の predictions にあるレースを race_management へ一括取り込み（Race ID重複はスキップ）。"""
    pred = read_sheet_df(spreadsheet, PREDICTIONS_SHEET, fresh=True)
    if pred.empty or not all(c in pred.columns for c in ("Race Day", "Venue", "Race Name", "Distance", "Horse Name")):
        return 0
    clean = ensure_21_columns(pred)
    clean = clean[(clean["Race Day"].str.strip() != "") & (clean["Venue"].str.strip() != "") & (clean["Race Name"].str.strip() != "")]
    if clean.empty:
        return 0
    clean = clean.assign(_rid=clean.apply(_race_id_from_row, axis=1))
    ws = ensure_management_sheet(spreadsheet)
    values = ws.get_all_values()
    known = {str(r[0]).strip() for r in values[1:] if r and str(r[0]).strip()}
    rows = []
    for rid, g in clean.groupby("_rid", sort=False):
        if not rid or rid in known:
            continue
        first = g.iloc[0]
        status = STATUS_RESULT if g["Placing"].astype(str).str.strip().ne("").any() else STATUS_PREDICTED
        rows.append([rid, first["Race Day"], first["Venue"], first["Race Name"], _clean_num(first["Distance"]),
                     status, str(len(g)), now_str(), now_str()])
        known.add(rid)
    if rows:
        ws.append_rows(rows, value_input_option="USER_ENTERED", table_range="A1")
        clear_sheet_cache()
    return len(rows)


# ============================================================
# 21列データの整形・前処理（既存ロジックを維持）
# ============================================================
def ensure_21_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in COLUMNS_21:
        if col not in df.columns:
            if col == "Distance":
                df[col] = "1800"
            elif col == "Burden Weight":
                df[col] = "55.0"
            elif col == "Track_Type":
                df[col] = "1"
            elif col == "AI_Probability":
                df[col] = "0.0"
            elif col == "AI_Rank":
                df[col] = "1"
            else:
                df[col] = ""
    return df[COLUMNS_21].fillna("").astype(str).copy()


def parse_input_table(raw_text: str) -> pd.DataFrame:
    """21列のCSV/TSVテキストを読み込み、統一した列順のDataFrameを返す。"""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("入力データが空です。競馬データを貼り付けてください。")

    text = raw_text.strip().lstrip("\ufeff")
    first_line = text.splitlines()[0]
    delimiter = "\t" if "\t" in first_line else ","

    try:
        parsed = pd.read_csv(
            StringIO(text), sep=delimiter, header=None, dtype=str,
            keep_default_na=False, skip_blank_lines=True, engine="python",
        )
    except Exception as exc:
        raise ValueError(f"CSV/TSVを解析できませんでした。区切り文字や引用符を確認してください。詳細: {exc}") from exc

    if parsed.empty:
        raise ValueError("読み取れるデータ行がありません。")

    first_values = [str(v).strip().lstrip("\ufeff") for v in parsed.iloc[0].tolist()]
    has_header = len(first_values) == len(COLUMNS_21) and [norm(v) for v in first_values] == [norm(v) for v in COLUMNS_21]
    if has_header:
        parsed = parsed.iloc[1:].reset_index(drop=True)

    if parsed.empty:
        raise ValueError("ヘッダー行しかありません。競走馬のデータ行も貼り付けてください。")

    if len(parsed.columns) != len(COLUMNS_21):
        raise ValueError(
            f"列数が不正です。21列必要ですが、{len(parsed.columns)}列検出されました。"
            "カンマ区切りCSVまたはタブ区切りTSVを貼り付けてください。"
        )

    parsed.columns = COLUMNS_21
    return ensure_21_columns(parsed)


def parse_tactics(v):
    if v is None or (not isinstance(v, str) and pd.isna(v)):
        return np.nan
    s = str(v).strip()
    m = re.match(r"^([1-4])(?:\.0+)?$", s)
    if m:
        return float(m.group(1))
    for key, code in (("逃", 1), ("先", 2), ("差", 3), ("追", 4)):
        if key in s:
            return float(code)
    return np.nan


def to_number(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(s.str.extract(r"(-?\d+(?:\.\d+)?)")[0], errors="coerce")


def preprocess(df: pd.DataFrame, default_values: dict) -> pd.DataFrame:
    df = df.copy()
    for k, v in default_values.items():
        if k in df.columns:
            df[k] = df[k].replace("", str(v)).fillna(str(v))

    for c in NUM_DEFAULTS:
        if c in df.columns:
            df[c] = to_number(df[c])
            rule = NUM_DEFAULTS[c]
            if isinstance(rule, str):
                kind, fallback = rule.split(":")
                base = df[c].median() if kind == "median" else df[c].mean()
                fill = base if pd.notna(base) else float(fallback)
            else:
                fill = rule
            df[c] = df[c].fillna(fill).astype(float)

    code = df["Tactics"].map(parse_tactics)
    df["TacticsCode"] = code
    df["脚質ラベル"] = code.map(lambda x: TACTICS_LABEL.get(int(x), "不明") if pd.notna(x) else "不明")
    numeric_code = pd.to_numeric(code, errors="coerce")
    med = numeric_code.median()
    df["Tactics"] = numeric_code.fillna(med if pd.notna(med) else 2.0).astype(float)

    for c in STR_COLS:
        if c in df.columns:
            df[c] = df[c].fillna("不明").astype(str).str.strip()

    if (df["Horse Number"] == 0).all():
        df["Horse Number"] = np.arange(1, len(df) + 1, dtype=float)

    return df


# ============================================================
# 展開予想・脚質分布（ペース診断）
# ============================================================
def diagnose_pace_from_codes(codes: pd.Series, total: int) -> dict:
    """前走脚質コード(1〜4)の頭数分布と、逃げ・先行割合に基づくペース診断。"""
    code = pd.to_numeric(codes, errors="coerce").dropna()
    known = len(code)
    n_nige = int((code == 1).sum())
    n_senko = int((code == 2).sum())
    n_sashi = int((code == 3).sum())
    n_oikomi = int((code == 4).sum())
    n_unknown = int(total - known)
    base = {
        "total": int(total), "known": known, "n_nige": n_nige, "n_senko": n_senko,
        "n_sashi": n_sashi, "n_oikomi": n_oikomi, "n_unknown": n_unknown,
    }
    if known == 0:
        return {"ok": False, **base}

    front_ratio = (n_nige + n_senko) / known
    # 逃げ・先行の割合と逃げ馬の頭数からハイペース確率を推定
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
        "front_ratio": round(front_ratio * 100, 1), **base,
    }


def diagnose_pace(df: pd.DataFrame) -> dict:
    return diagnose_pace_from_codes(df["TacticsCode"], len(df))


def render_pace(p: dict, title: str = "③ 展開予想・脚質分布（ペース診断）"):
    st.subheader(title)
    if not p.get("ok"):
        st.info("前走脚質（Tactics）が1頭も読み取れなかったため、脚質分布・ペース診断は行えません。")
        return

    st.caption(f"前走脚質が判明している {p['known']} / {p['total']} 頭を集計しています。")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🏃 逃げ", f"{p['n_nige']}頭")
    c2.metric("🐎 先行", f"{p['n_senko']}頭")
    c3.metric("🏇 差し", f"{p['n_sashi']}頭")
    c4.metric("💨 追込", f"{p['n_oikomi']}頭")
    c5.metric("❔ 不明", f"{p['n_unknown']}頭")

    with st.container(border=True):
        st.markdown("**🧭 ペース診断カード**")
        m1, m2, m3 = st.columns(3)
        m1.metric("想定ペース", p["pace"])
        m2.metric("ハイペース確率", f"{p['high_prob']:.1f}%")
        m3.metric("逃げ・先行の割合", f"{p['front_ratio']:.1f}%")
        front = p["n_nige"] + p["n_senko"]
        if p["pace"] == "ハイペース":
            st.error(f"逃げ・先行馬が {front} 頭（{p['front_ratio']:.1f}%）と多く、前が競り合うハイペース想定です。差し・追込が有利になりやすい展開です。")
        elif p["pace"] == "スローペース":
            st.success(f"逃げ・先行馬は {front} 頭（{p['front_ratio']:.1f}%）と少なく、楽に運べるスローペース想定です。逃げ・先行が有利になりやすい展開です。")
        else:
            st.warning(f"逃げ・先行馬は {front} 頭（{p['front_ratio']:.1f}%）で、平均的なミドルペース想定です。")


# ============================================================
# モデル読み込み・予測（既存ロジックを維持）
# ============================================================
@st.cache_resource(show_spinner="AIの頭脳を読み込み中…")
def load_model():
    if not MODEL_PATH.exists():
        return None, {}
    with open(MODEL_PATH, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and isinstance(obj.get("models"), dict):
        if not all(k in obj["models"] for k in ("win", "top3")):
            raise ValueError("model.pkl の models に win / top3 がありません。")
        return obj, {"bundle": obj, "features": list(obj.get("features", [])), "encoders": obj.get("encoders", {})}

    meta = {"features": None, "encoders": {}}
    model = obj
    if isinstance(obj, dict):
        model = next((obj[k] for k in ("model", "pipeline", "clf", "estimator", "booster", "lgbm", "xgb") if k in obj), None)
        if model is None:
            model = next((v for v in obj.values() if hasattr(v, "predict")), None)
        for k in ("features", "feature_names", "feature_cols", "feature_columns", "columns"):
            if k in obj and obj[k] is not None:
                meta["features"] = list(obj[k]); break
        for k in ("encoders", "label_encoders", "encoder", "le_dict"):
            if k in obj and isinstance(obj[k], dict):
                meta["encoders"] = obj[k]; break
    elif isinstance(obj, (tuple, list)):
        model = next((v for v in obj if hasattr(v, "predict")), None)
        for v in obj:
            if isinstance(v, dict) and not meta["encoders"]: meta["encoders"] = v
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v): meta["features"] = list(v)
    return model, meta


def _load_spreadsheet_history(spreadsheet):
    """学習・特徴量用の過去データ。出走表(バッファ)や管理用シートは含めない。"""
    frames = []
    for title in _cached_worksheet_titles(spreadsheet.id):
        if title in SYSTEM_SHEETS:
            continue
        ws = spreadsheet.worksheet(title)
        try:
            part = _read_worksheet_dataframe(ws)
        except Exception as exc:
            raise RuntimeError(f"スプレッドシートの「{ws.title}」タブを読み込めません: {exc}") from exc
        if part.empty or "Horse Name" not in part.columns:
            continue
        if not ("Race Day" in part.columns or "Placing" in part.columns):
            continue
        frames.append(ensure_21_columns(part))
    if not frames:
        return pd.DataFrame(columns=COLUMNS_21)
    return pd.concat(frames, ignore_index=True)


def _new_model_matrix(df, bundle):
    features = list(bundle["features"])
    numeric = set(bundle.get("numeric_features", []))
    categorical = set(bundle.get("categorical_features", []))
    encoders = bundle.get("encoders", {})
    spreadsheet = get_gspread_client()
    hist = _load_spreadsheet_history(spreadsheet)
    hist = ensure_21_columns(hist).drop_duplicates(
        subset=["Race Day", "Race Name", "Venue", "Horse Number", "Horse Name"], keep="last"
    )
    hist["_day"] = pd.to_datetime(hist["Race Day"], errors="coerce")
    hist["_placing"] = to_number(hist["Placing"])
    hist["_last3f"] = to_number(hist["Last 3F"])
    hist["_tactics"] = hist["Tactics"].map(parse_tactics)

    try:
        horses_ws = spreadsheet.worksheet(HORSES_SHEET)
    except Exception as exc:
        raise RuntimeError("Googleスプレッドシートに「horses_master」タブがありません。") from exc
    shorses = _read_worksheet_dataframe(horses_ws)
    pedigree = {}
    if not shorses.empty and "Horse Name" in shorses.columns:
        for _, hr in shorses.fillna("").iterrows():
            pedigree[_norm_horse(hr.get("Horse Name", ""))] = (hr.get("Sire", ""), hr.get("Dam's Sire", ""))

    rows = []
    for _, r in df.iterrows():
        horse_key = _norm_horse(r.get("Horse Name", ""))
        race_day = pd.to_datetime(r.get("Race Day", ""), errors="coerce")
        past = hist[(hist["Horse Name"].map(_norm_horse) == horse_key) & hist["_day"].notna() & (hist["_day"] < race_day)]
        past = past.sort_values("_day")
        last3 = past.tail(3)
        placings = past["_placing"].dropna()
        last3f = last3["_last3f"].dropna()
        prev3f = past["_last3f"].dropna()
        sire, damsire = pedigree.get(horse_key, ("Unknown", "Unknown"))
        row = {
            "Distance": r.get("Distance", 1800), "Bracket": r.get("Bracket", 0),
            "Horse Number": r.get("Horse Number", 0), "Horse Age": r.get("Horse Age", 4),
            "Burden Weight": r.get("Burden Weight", 55), "Track_Type": r.get("Track_Type", 1),
            "Hist_Starts": float(len(placings)),
            "Hist_Win_Rate": float((placings == 1).mean()) if len(placings) else 0.0,
            "Hist_Top3_Rate": float((placings <= 3).mean()) if len(placings) else 0.0,
            "Hist_Mean_Placing3": float(last3["_placing"].dropna().mean()) if last3["_placing"].notna().any() else 0.0,
            "Hist_Last3F_Mean3": float(last3f.mean()) if len(last3f) else 0.0,
            "Hist_Last3F_Last": float(prev3f.iloc[-1]) if len(prev3f) else 0.0,
            # 入力された前走脚質を優先。空欄なら履歴の最新脚質、なければ0（不明）
            "Hist_Last_Tactics": (
                float(parse_tactics(r.get("Previous_Tactics_Input")))
                if pd.notna(parse_tactics(r.get("Previous_Tactics_Input")))
                else (float(past["_tactics"].dropna().iloc[-1]) if past["_tactics"].notna().any() else 0.0)
            ),
            "Race Name": r.get("Race Name", "Unknown"), "Venue": r.get("Venue", "Unknown"),
            "Track Condition": r.get("Track Condition", "Unknown"), "Weather": r.get("Weather", "Unknown"),
            "Horse Name": r.get("Horse Name", "Unknown"), "Sex": r.get("Sex", "Unknown"),
            "Jockey": r.get("Jockey", "Unknown"), "Trainer": r.get("Trainer", "Unknown"),
            "Sire": sire or "Unknown", "Dam's Sire": damsire or "Unknown",
        }
        rows.append(row)
    X = pd.DataFrame(rows, columns=features)
    for c in features:
        if c in categorical:
            mapping = encoders.get(c, {})
            X[c] = X[c].fillna("Unknown").astype(str).map(mapping).fillna(-1).astype("int32")
        elif c in numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0).astype(float)
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)
    return X


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


def apply_encoder(series, enc):
    if isinstance(enc, dict):
        mapping = enc
    elif hasattr(enc, "classes_"):
        mapping = {v: i for i, v in enumerate(enc.classes_)}
    else:
        return None
    return series.map(mapping).fillna(-1).astype(float)


def encode_X(X, model, meta):
    warns = []
    cat_feats = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype).startswith("string")]
    if hasattr(model, "steps"):
        return X, warns

    for c in cat_feats:
        enc = meta["encoders"].get(c) if meta["encoders"] else None
        coded = apply_encoder(X[c], enc) if enc is not None else None
        if coded is None:
            coded = X[c].map(lambda v: float(zlib.crc32(str(v).encode()) % 1000))
            warns.append(c)
        X[c] = coded
    return X, warns


def predict_top3(model, X) -> np.ndarray:
    X_val = X.values if hasattr(X, "values") else X
    try:
        preds = model.predict(X_val, predict_disable_shape_check=True)
    except TypeError:
        preds = model.predict(X_val)
    return np.clip(np.asarray(preds, dtype=float), 0, 1)


def get_feature_names_custom(model, meta):
    if meta.get("features"):
        return meta["features"]
    for attr in ("feature_names_in_", "feature_name_", "feature_names_", "feature_names"):
        v = getattr(model, attr, None)
        if v is not None and not callable(v):
            try:
                return list(v)
            except TypeError:
                pass
    return None


def run_prediction_for_race(model, meta, race_df: pd.DataFrame):
    """
    1レース分の出走表（21列 + Previous_Tactics_Input等）からAI予想を実行する。
    戻り値: (out_21: 21列データ, view_df: 表示用DataFrame, pace: ペース診断dict)
    """
    df = preprocess(race_df, {"Execution Date": now_str()})
    pace = diagnose_pace(df)
    top3_pct = None
    if isinstance(model, dict) and isinstance(model.get("models"), dict):
        X = _new_model_matrix(df, model)
        win_model = model["models"]["win"]
        top3_model = model["models"]["top3"]
        proba = np.asarray(win_model.predict(X, num_iteration=getattr(win_model, "best_iteration", None)), dtype=float)
        top3_proba = np.asarray(top3_model.predict(X, num_iteration=getattr(top3_model, "best_iteration", None)), dtype=float)
        if len(proba) != len(df) or len(top3_proba) != len(df):
            raise ValueError("モデルの予測件数と出馬表の頭数が一致しません。")
        top3_pct = np.round(np.clip(top3_proba, 0, 1) * 100, 1)
    else:
        feats = get_feature_names_custom(model, meta) or ["Bracket", "Horse Number", "Horse Age", "Burden Weight", "Tactics", "Distance", "Last 3F"]
        X, _missing = build_X(df, feats)
        X, _hashed = encode_X(X, model, meta)
        proba = predict_top3(model, X)

    proba_pct = np.round(np.asarray(proba, dtype=float) * 100, 1)
    df["AI_Probability"] = proba_pct.astype(str)
    df["AI_Rank"] = pd.Series(proba_pct, index=df.index).rank(ascending=False, method="min").astype(int).astype(str)
    if top3_pct is not None:
        df["複勝確率(3着以内)"] = top3_pct.astype(str)
    df_sorted = df.sort_values("AI_Probability", key=lambda col: pd.to_numeric(col, errors="coerce"), ascending=False).reset_index(drop=True)
    out_21 = ensure_21_columns(df_sorted)
    return out_21, df_sorted, pace


# ============================================================
# 保存ロジック（race_results / predictions / 出走表バッファ）
# ============================================================
RC = {c: i for i, c in enumerate(RESULT_COLUMNS)}


def _clean_num(v) -> str:
    """'3.0' -> '3'、数値でなければそのまま返す。"""
    s = str(v if v is not None else "").strip()
    if s == "":
        return ""
    try:
        f = float(s.replace(",", ""))
        return str(int(f)) if f == int(f) else s
    except Exception:
        return s


def _race_meta_from_row(row) -> dict:
    return {
        "Race Day": str(row.get("Race Day", "")).strip(),
        "Venue": str(row.get("Venue", "")).strip(),
        "Race Name": str(row.get("Race Name", "")).strip(),
        "Distance": _clean_num(row.get("Distance", "")),
    }


def _pad(row, n):
    row = list(row)
    return row + [""] * (n - len(row)) if len(row) < n else row[:n]


def _race_result_rows(values, rid):
    """race_results の全値から、指定 Race ID の (行番号, 行データ) 一覧を返す。"""
    out = []
    for rn, row in enumerate(values[1:], start=2):
        row = _pad(row, len(RESULT_COLUMNS))
        if str(row[RC["Race ID"]]).strip() == rid:
            out.append((rn, row))
    return out


def predictions_rows_for_race(spreadsheet, rid):
    """predictions から該当レースの (行番号リスト, 21列DataFrame) を最新値で取得する。"""
    try:
        ws = spreadsheet.worksheet(PREDICTIONS_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        return [], pd.DataFrame(columns=COLUMNS_21)
    values = ws.get_all_values()
    df = values_to_dataframe(values)
    if df.empty or "Horse Name" not in df.columns:
        return [], pd.DataFrame(columns=COLUMNS_21)
    header = [str(v).strip() for v in values[0]]
    nums, keep = [], []
    day_i = header.index("Race Day") if "Race Day" in header else None
    if day_i is None:
        return [], pd.DataFrame(columns=COLUMNS_21)
    hidx = {h: i for i, h in enumerate(header)}
    for rn, row in enumerate(values[1:], start=2):
        row = _pad(row, len(header))
        if not any(str(x).strip() for x in row):
            continue
        key = _race_id(row[hidx.get("Race Day", 0)], row[hidx.get("Venue", 0)] if "Venue" in hidx else "",
                       row[hidx.get("Race Name", 0)] if "Race Name" in hidx else "",
                       row[hidx.get("Distance", 0)] if "Distance" in hidx else "")
        if key == rid:
            nums.append(rn)
            keep.append(dict(zip(header, row)))
    if not keep:
        return [], pd.DataFrame(columns=COLUMNS_21)
    return nums, ensure_21_columns(pd.DataFrame(keep))


def save_prediction_snapshot(spreadsheet, rid, out_21, view_df, pace):
    """
    AI予想の結果を全て race_results に保存する（Race ID単位でupsert）。
    既に登録済みの実績（Tactics/Last 3F/Placing/Updated At）は馬名で引き継ぐ。
    """
    ws = ensure_results_sheet(spreadsheet)
    values = ws.get_all_values()
    old_rows = _race_result_rows(values, rid)
    keep_result = {}
    for _rn, row in old_rows:
        keep_result[_norm_horse(row[RC["Horse Name"]])] = {
            "Tactics": row[RC["Tactics"]], "Last 3F": row[RC["Last 3F"]],
            "Placing": row[RC["Placing"]], "Updated At": row[RC["Updated At"]],
        }

    pace_ok = bool(pace.get("ok"))
    predicted_at = now_str()
    new_rows = []
    out = out_21.reset_index(drop=True)
    view = view_df.reset_index(drop=True)
    for i in range(len(out)):
        r, v = out.iloc[i], view.iloc[i]
        kept = keep_result.get(_norm_horse(r["Horse Name"]), {})
        rec = {
            "Race ID": rid, "Race Day": str(r["Race Day"]).strip(), "Venue": str(r["Venue"]).strip(),
            "Race Name": str(r["Race Name"]).strip(), "Distance": _clean_num(r["Distance"]),
            "Horse Number": _clean_num(r["Horse Number"]), "Horse Name": str(r["Horse Name"]).strip(),
            "Tactics": kept.get("Tactics", ""), "Last 3F": kept.get("Last 3F", ""),
            "Placing": kept.get("Placing", ""), "Updated At": kept.get("Updated At", ""),
            "Execution Date": str(r["Execution Date"]), "Track Condition": str(r["Track Condition"]),
            "Weather": str(r["Weather"]), "Bracket": _clean_num(r["Bracket"]), "Sex": str(r["Sex"]),
            "Horse Age": _clean_num(r["Horse Age"]), "Burden Weight": str(r["Burden Weight"]),
            "Jockey": str(r["Jockey"]), "Trainer": str(r["Trainer"]), "Track_Type": _clean_num(r["Track_Type"]),
            "Input Tactics": _clean_num(v.get("_in_tactics", "")), "Input Last 3F": str(v.get("_in_last3f", "") or ""),
            "AI_Probability": str(r["AI_Probability"]), "AI_Rank": str(r["AI_Rank"]),
            "Top3 Probability": str(v.get("複勝確率(3着以内)", "") or ""),
            "Pace": pace.get("pace", "") if pace_ok else "",
            "High Pace Prob": str(pace.get("high_prob", "")) if pace_ok else "",
            "Front Ratio": str(pace.get("front_ratio", "")) if pace_ok else "",
            "Predicted At": predicted_at,
        }
        new_rows.append([str(rec.get(c, "")) for c in RESULT_COLUMNS])

    if not new_rows:
        raise ValueError("保存する予想結果がありません。")
    # 先に追記 → 成功後に旧行を削除（途中失敗でもデータが消えない順序）
    ws.append_rows(new_rows, value_input_option="USER_ENTERED", table_range="A1")
    _delete_rows_by_numbers(ws, [rn for rn, _ in old_rows])
    clear_sheet_cache()
    return len(new_rows)


def upsert_training_rows(spreadsheet, rid, df21):
    """
    predictions（学習用）へレース結果＋AI予想（勝率・予想順位のみ）を保存する。
    同一 Race ID の既存行は置き換える（重複蓄積を防止）。
    """
    try:
        ws = spreadsheet.worksheet(PREDICTIONS_SHEET)
    except gspread.exceptions.WorksheetNotFound:
        ws = spreadsheet.add_worksheet(title=PREDICTIONS_SHEET, rows=5000, cols=len(COLUMNS_21))
        ws.update(range_name="A1", values=[COLUMNS_21], value_input_option="USER_ENTERED")
    values = ws.get_all_values()
    header = [str(v).strip() for v in values[0]] if values else []
    while header and not header[-1]:
        header.pop()
    if not header:
        ws.update(range_name="A1", values=[COLUMNS_21], value_input_option="USER_ENTERED")
        header, values = list(COLUMNS_21), [list(COLUMNS_21)]
    missing = [c for c in COLUMNS_21 if c not in header]
    if missing:
        raise ValueError(f"「{PREDICTIONS_SHEET}」タブのヘッダーに不足があります（{', '.join(missing)}）。既存データ保護のため自動変更しません。")

    hidx = {h: i for i, h in enumerate(header)}
    del_rows = []
    for rn, row in enumerate(values[1:], start=2):
        row = _pad(row, len(header))
        if _race_id(row[hidx["Race Day"]], row[hidx["Venue"]], row[hidx["Race Name"]], row[hidx["Distance"]]) == rid:
            del_rows.append(rn)

    new_rows = [[str(rec.get(h, "")) if h in COLUMNS_21 else "" for h in header] for rec in df21.to_dict("records")]
    if new_rows:
        ws.append_rows(new_rows, value_input_option="USER_ENTERED", table_range="A1")
    _delete_rows_by_numbers(ws, del_rows)
    clear_sheet_cache()
    return len(new_rows), len(del_rows)


def register_results(spreadsheet, race_row, result_df):
    """
    実績登録:
      1) race_results の該当行へ実績（Tactics/Last 3F/Placing）を反映（AI予想の全項目は保持）
      2) predictions へ「レース結果 + AI予想（勝率・予想順位のみ）」を保存
      3) race_management のステータスを「結果登録済み」へ更新
    """
    rid = str(race_row["Race ID"])
    meta = _race_meta_from_row(race_row)
    ws = ensure_results_sheet(spreadsheet)
    values = ws.get_all_values()
    existing = _race_result_rows(values, rid)
    by_name = {_norm_horse(r[RC["Horse Name"]]): (rn, r) for rn, r in existing}
    by_num = {_clean_num(r[RC["Horse Number"]]): (rn, r) for rn, r in existing}

    # 旧データ（predictionsのみに予想がある）用のAI予想フォールバック
    _nums, pred_old = predictions_rows_for_race(spreadsheet, rid)
    pred_by_name = {_norm_horse(r["Horse Name"]): r for _, r in pred_old.iterrows()} if not pred_old.empty else {}

    now = now_str()
    updates, appends, unmatched_by_number = [], [], []
    for _, rr in result_df.iterrows():
        name, num = str(rr["Horse Name"]).strip(), _clean_num(rr["Horse Number"])
        if not name or not num:
            continue
        vals = [str(rr["Tactics"]).strip(), str(rr["Last 3F"]).strip(), str(rr["Placing"]).strip(), now]
        hit = by_name.get(_norm_horse(name))
        if hit is None and num in by_num:
            hit = by_num[num]
            unmatched_by_number.append(name)
        if hit is not None:
            updates.append({"range": f"H{hit[0]}:K{hit[0]}", "values": [vals]})
            continue
        p = pred_by_name.get(_norm_horse(name))
        rec = {c: "" for c in RESULT_COLUMNS}
        rec.update({
            "Race ID": rid, "Race Day": meta["Race Day"], "Venue": meta["Venue"], "Race Name": meta["Race Name"],
            "Distance": meta["Distance"], "Horse Number": num, "Horse Name": name,
            "Tactics": vals[0], "Last 3F": vals[1], "Placing": vals[2], "Updated At": now,
        })
        if p is not None:
            for c in ("Execution Date", "Track Condition", "Weather", "Bracket", "Sex", "Horse Age",
                      "Burden Weight", "Jockey", "Trainer", "Track_Type", "AI_Probability", "AI_Rank"):
                rec[c] = str(p.get(c, ""))
        appends.append([str(rec[c]) for c in RESULT_COLUMNS])

    if not updates and not appends:
        raise ValueError("保存対象がありません。馬番と馬名を確認してください。")
    if updates:
        ws.batch_update(updates, value_input_option="USER_ENTERED")
    if appends:
        ws.append_rows(appends, value_input_option="USER_ENTERED", table_range="A1")
    clear_sheet_cache()

    # 最新の race_results から predictions 用の学習データを組み立てる
    fresh = _race_result_rows(ws.get_all_values(), rid)
    train = []
    for _rn, r in fresh:
        placing = str(r[RC["Placing"]]).strip()
        if not placing:
            continue
        train.append({
            "Execution Date": r[RC["Execution Date"]] or r[RC["Predicted At"]] or now,
            "Race Day": r[RC["Race Day"]], "Race Name": r[RC["Race Name"]], "Venue": r[RC["Venue"]],
            "Distance": r[RC["Distance"]], "Track Condition": r[RC["Track Condition"]], "Weather": r[RC["Weather"]],
            "Bracket": r[RC["Bracket"]], "Horse Number": r[RC["Horse Number"]], "Horse Name": r[RC["Horse Name"]],
            "Sex": r[RC["Sex"]], "Horse Age": r[RC["Horse Age"]], "Burden Weight": r[RC["Burden Weight"]],
            "Jockey": r[RC["Jockey"]], "Trainer": r[RC["Trainer"]],
            "Tactics": r[RC["Tactics"]], "Last 3F": r[RC["Last 3F"]],
            "AI_Probability": r[RC["AI_Probability"]], "AI_Rank": r[RC["AI_Rank"]],
            "Placing": placing, "Track_Type": r[RC["Track_Type"]] or "1",
        })
    n_train, n_replaced = (0, 0)
    if train:
        n_train, n_replaced = upsert_training_rows(spreadsheet, rid, pd.DataFrame(train, columns=COLUMNS_21))
    upsert_race_management(spreadsheet, rid, meta, STATUS_RESULT, horse_count=len(fresh))
    return {"updated": len(updates), "appended": len(appends), "train_rows": n_train,
            "train_replaced": n_replaced, "matched_by_number": unmatched_by_number}


def previous_result_lookup(spreadsheet, race_day):
    """対象レース日より前の、各馬の最新実績（脚質・上がり3F）を race_results から取得する。"""
    df = read_sheet_df(spreadsheet, RESULTS_SHEET, fresh=True)
    if df.empty or not all(c in df.columns for c in ("Race Day", "Horse Name", "Tactics", "Last 3F")):
        return {}
    df = df[df["Tactics"].astype(str).str.strip() != ""].copy()
    if df.empty:
        return {}
    target = pd.to_datetime(race_day, errors="coerce")
    df["_date"] = pd.to_datetime(df["Race Day"], errors="coerce")
    if pd.notna(target):
        df = df[df["_date"] < target]
    df["_horse"] = df["Horse Name"].map(_norm_horse)
    df = df.sort_values("_date")
    lookup = {}
    for _, row in df.iterrows():
        if row["_horse"]:
            lookup[row["_horse"]] = {"Tactics": parse_tactics(row.get("Tactics", "")), "Last 3F": row.get("Last 3F", "")}
    return lookup


# ============================================================
# 出走表（バッファ）の読み込み・訂正・クリア
# ============================================================
def load_entry_buffer(spreadsheet):
    """出走表を読み込み、Race IDごとに race_management へ『未予想』で登録（重複しない）。"""
    ensure_entry_sheet(spreadsheet)
    raw = read_sheet_df(spreadsheet, ENTRY_SHEET, fresh=True)
    if raw.empty or "Horse Name" not in raw.columns:
        return pd.DataFrame(columns=COLUMNS_21), []
    df = ensure_21_columns(raw)
    df = df[df["Horse Name"].str.strip() != ""].reset_index(drop=True)
    if df.empty:
        return df, []
    rids = []
    for rid, g in df.assign(_rid=df.apply(_race_id_from_row, axis=1)).groupby("_rid", sort=False):
        if not rid:
            continue
        upsert_race_management(spreadsheet, rid, _race_meta_from_row(g.iloc[0]), STATUS_UNPREDICTED, len(g))
        rids.append(rid)
    return df, rids


def clear_predicted_from_buffer(spreadsheet, predicted_rids):
    """
    予想が保存されたレースの行を出走表から取り除き、ヘッダーのみ残す。
    （未予想の別レースが残っている場合のみ、その行を書き戻す）
    """
    raw = read_sheet_df(spreadsheet, ENTRY_SHEET, fresh=True)
    remain = pd.DataFrame(columns=COLUMNS_21)
    if not raw.empty and "Horse Name" in raw.columns:
        d = ensure_21_columns(raw)
        d = d[d["Horse Name"].str.strip() != ""]
        keep = d[~d.apply(_race_id_from_row, axis=1).isin(set(predicted_rids))]
        remain = keep
    clear_entry_buffer(spreadsheet)
    if not remain.empty:
        write_entry_buffer(spreadsheet, remain, replace=False)
    return len(remain)


def load_race_for_correction(spreadsheet, rid):
    """過去データ（race_results、無ければpredictions）から指定Race IDの出走表を復元する。"""
    res = read_sheet_df(spreadsheet, RESULTS_SHEET, fresh=True)
    note = ""
    if not res.empty and "Race ID" in res.columns:
        sub = res[res["Race ID"].astype(str).str.strip() == rid].copy()
    else:
        sub = pd.DataFrame()
    if not sub.empty:
        for c in RESULT_COLUMNS:
            if c not in sub.columns:
                sub[c] = ""
        entry = pd.DataFrame({
            "Execution Date": "", "Race Day": sub["Race Day"], "Race Name": sub["Race Name"], "Venue": sub["Venue"],
            "Distance": sub["Distance"], "Track Condition": sub["Track Condition"], "Weather": sub["Weather"],
            "Bracket": sub["Bracket"], "Horse Number": sub["Horse Number"], "Horse Name": sub["Horse Name"],
            "Sex": sub["Sex"], "Horse Age": sub["Horse Age"], "Burden Weight": sub["Burden Weight"],
            "Jockey": sub["Jockey"], "Trainer": sub["Trainer"],
            "Tactics": sub["Input Tactics"], "Last 3F": sub["Input Last 3F"],
            "AI_Probability": "", "AI_Rank": "", "Placing": "", "Track_Type": sub["Track_Type"].replace("", "1"),
        })
    else:
        _nums, pred = predictions_rows_for_race(spreadsheet, rid)
        if pred.empty:
            return pd.DataFrame(columns=COLUMNS_21), "指定された Race ID のデータが race_results / predictions に見つかりません。"
        entry = pred.copy()
        entry["Tactics"], entry["Last 3F"] = "", ""
        entry["AI_Probability"], entry["AI_Rank"], entry["Placing"] = "", "", ""
        note = "race_results に予想が無かったため predictions から復元しました。前走脚質・上がり3Fは空欄です。入力し直してください。"
    entry = ensure_21_columns(entry)
    order = to_number(entry["Horse Number"]).fillna(999).sort_values(kind="stable").index
    return entry.loc[order].reset_index(drop=True), note


# ============================================================
# AI予想 vs 実績の比較表
# ============================================================
COMPARE_COLUMNS = [
    "Race ID", "Race Day", "Venue", "Race Name", "Horse Number", "Horse Name",
    "AI予想順位", "AI勝率(%)", "複勝確率(%)", "確定着順", "上がり3F(タイム)", "順位差(着順−AI順位)", "AI上位3頭の的中",
]


def _finish_compare(df: pd.DataFrame) -> pd.DataFrame:
    ai_rank = pd.to_numeric(df["AI予想順位"], errors="coerce")
    placing = to_number(df["確定着順"].astype(str))
    diff = placing - ai_rank
    df["順位差(着順−AI順位)"] = diff.map(lambda x: "" if pd.isna(x) else str(int(x)))
    hit = []
    for a, p in zip(ai_rank, placing):
        if pd.isna(a) or pd.isna(p):
            hit.append("")
        elif a <= 3:
            hit.append("○" if p <= 3 else "×")
        else:
            hit.append("")
    df["AI上位3頭の的中"] = hit
    df["_p"] = placing.fillna(999)
    df["_a"] = ai_rank.fillna(999)
    df = df.sort_values(["_p", "_a"], kind="stable").drop(columns=["_p", "_a"]).reset_index(drop=True)
    return df[COMPARE_COLUMNS]


def build_comparison_from_results(res_df: pd.DataFrame, rid: str) -> pd.DataFrame:
    sub = res_df[res_df["Race ID"].astype(str).str.strip() == rid].copy() if not res_df.empty and "Race ID" in res_df.columns else pd.DataFrame()
    if sub.empty:
        return pd.DataFrame(columns=COMPARE_COLUMNS)
    for c in RESULT_COLUMNS:
        if c not in sub.columns:
            sub[c] = ""
    out = pd.DataFrame({
        "Race ID": sub["Race ID"], "Race Day": sub["Race Day"], "Venue": sub["Venue"], "Race Name": sub["Race Name"],
        "Horse Number": sub["Horse Number"], "Horse Name": sub["Horse Name"],
        "AI予想順位": sub["AI_Rank"], "AI勝率(%)": sub["AI_Probability"], "複勝確率(%)": sub["Top3 Probability"],
        "確定着順": sub["Placing"], "上がり3F(タイム)": sub["Last 3F"],
    })
    return _finish_compare(out.reset_index(drop=True))


def build_comparison_from_predictions(pred_df: pd.DataFrame, rid: str) -> pd.DataFrame:
    """race_results に無い旧データ用：predictions（AI予想＋実績が入った行）から比較表を作る。"""
    if pred_df is None or pred_df.empty or "Race Day" not in pred_df.columns:
        return pd.DataFrame(columns=COMPARE_COLUMNS)
    d = ensure_21_columns(pred_df)
    d = d[d.apply(_race_id_from_row, axis=1) == rid]
    if d.empty:
        return pd.DataFrame(columns=COMPARE_COLUMNS)
    out = pd.DataFrame({
        "Race ID": rid, "Race Day": d["Race Day"], "Venue": d["Venue"], "Race Name": d["Race Name"],
        "Horse Number": d["Horse Number"], "Horse Name": d["Horse Name"],
        "AI予想順位": d["AI_Rank"], "AI勝率(%)": d["AI_Probability"], "複勝確率(%)": "",
        "確定着順": d["Placing"], "上がり3F(タイム)": d["Last 3F"],
    })
    return _finish_compare(out.reset_index(drop=True))


# ============================================================
# 表示ヘルパー
# ============================================================
def render_cards_prediction(df: pd.DataFrame):
    for _, row in df.reset_index(drop=True).iterrows():
        rank = row.get("AI_Rank", "-")
        prob = row.get("AI_Probability", "-")
        horse = row.get("Horse Name", "-")
        h_num = _clean_num(row.get("Horse Number", "-"))
        bracket = _clean_num(row.get("Bracket", "-"))
        jockey = row.get("Jockey", "-")
        trainer = row.get("Trainer", "-")
        sex = row.get("Sex", "")
        age = _clean_num(row.get("Horse Age", ""))
        weight = row.get("Burden Weight", "")
        tactics = row.get("脚質ラベル", row.get("Tactics", ""))
        last3f = row.get("Last 3F", "")

        badge = "🥇" if str(rank) == "1" else ("🥈" if str(rank) == "2" else ("🥉" if str(rank) == "3" else "🏇"))

        with st.container(border=True):
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.markdown(f"### {badge} {h_num}番 **{horse}**")
                st.caption(f"{bracket}枠 | {sex}{age} | 斤量: {weight}kg | {jockey} ({trainer})")
                st.text(f"前走脚質: {tactics} | 前走上がり3F: {last3f}s")
            with col_right:
                st.metric("AI順位", f"{rank}位")
                st.metric("勝率", f"{prob}%")
                if "複勝確率(3着以内)" in row.index:
                    st.metric("複勝確率（3着以内）", f"{row.get('複勝確率(3着以内)', '-')}%")


def render_cards_history(df: pd.DataFrame):
    df_render = df.reset_index(drop=True).head(50)
    if len(df) > 50:
        st.info(f"💡 データが全 {len(df)} 件あるため、先頭の 50 件を表示しています。")

    for _, row in df_render.iterrows():
        r_name = row.get("Race Name", "レース")
        r_day = row.get("Race Day", "")
        venue = row.get("Venue", "")
        dist = row.get("Distance", "")
        horse = row.get("Horse Name", "")
        h_num = row.get("Horse Number", "")
        rank = row.get("AI_Rank", "-")
        prob = row.get("AI_Probability", "-")
        placing = row.get("Placing", "-")

        badge = "🥇" if str(rank) == "1" else ("🥈" if str(rank) == "2" else ("🥉" if str(rank) == "3" else "🐴"))

        with st.container(border=True):
            st.markdown(f"**{r_day} {venue} {r_name} ({dist}m)**")
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f"{badge} **{h_num}番 {horse}**")
                st.caption(f"騎手: {row.get('Jockey', '-')} / 斤量: {row.get('Burden Weight', '-')}kg")
            with c2:
                st.metric("AI予想", f"{rank}位", f"{prob}%")
            with c3:
                st.metric("実着順", f"{placing}着" if placing else "-")


def csv_download_button(df: pd.DataFrame, filename: str, label: str, key: str):
    st.download_button(label, data=to_csv_bytes(df), file_name=filename, mime="text/csv", key=key, **STRETCH)


def set_entry(df: pd.DataFrame, source: str, keep_results: bool = False):
    """作業中の出走表をセッションに保持（エディタはバージョン更新でリセット）。"""
    st.session_state["entry_df"] = df.reset_index(drop=True)
    st.session_state["entry_source"] = source
    st.session_state["entry_version"] = st.session_state.get("entry_version", 0) + 1
    if not keep_results:
        st.session_state.pop("results", None)
        st.session_state.pop("save_notice", None)


def flash(kind: str, msg: str):
    st.session_state["_flash"] = (kind, msg)


def show_flash():
    f = st.session_state.pop("_flash", None)
    if f:
        getattr(st, f[0])(f[1])


def _blank(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


RESULT_ALIASES = {
    "馬番": "Horse Number", "馬名": "Horse Name", "脚質": "Tactics", "前走脚質": "Tactics",
    "上がり3F": "Last 3F", "上がり": "Last 3F", "着順": "Placing", "順位": "Placing",
}
RESULT_REQUIRED = ["Horse Number", "Horse Name", "Tactics", "Last 3F", "Placing"]


def _do_register(sh, chosen, result_df: pd.DataFrame):
    """実績登録の共通処理（エラーは呼び出し側へ）。"""
    result_df = result_df[RESULT_REQUIRED].fillna("").astype(str)
    result_df = result_df[(result_df["Horse Number"].str.strip() != "") & (result_df["Horse Name"].str.strip() != "")]
    if result_df.empty:
        raise ValueError("馬番と馬名が入力された行がありません。")
    if result_df["Placing"].str.strip().eq("").all():
        raise ValueError("着順が1頭も入力されていません。")
    return register_results(sh, chosen, result_df)


# ============================================================
# アプリケーション本体
# ============================================================
st.title("🐴 自分専用 競馬予想AI（オールインワン版）")

try:
    model, meta = load_model()
    if model is None:
        st.error("model.pkl が見つからないか、有効なモデルではありません。")
        st.stop()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

st.caption("スプレッドシートは各画面の「読み込む」ボタンを押したときだけ取得します（起動時の自動読み込みはありません）。")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 出走表・AI予想",
    "📊 AI予想 vs 実績",
    "🏁 実績登録・一括登録",
    "🗂️ レース管理",
    "🧬 血統データ登録",
])

# ------------------------------------------------------------
# TAB 1: 出走表（バッファ）→ AI予想 → 自動クリア ／ 予想訂正
# ------------------------------------------------------------
with tab1:
    show_flash()
    st.subheader("① 出走表の準備（「出走表」タブ = 一時バッファ）")
    st.caption("予想結果の保存が完了すると、出走表タブのデータ行（2行目以降）は自動でクリアされ、ヘッダーのみが残ります。")

    with st.expander("📝 出走表を新規入力・アップロード（「出走表」タブへ書き込み）", expanded=False):
        up = st.file_uploader("CSV / TSV ファイル（21列）", type=["csv", "tsv", "txt"], key="entry_upload")
        raw_entry = st.text_area("または 21列の TSV / CSV を貼り付け", height=160, key="entry_text", placeholder="\t".join(COLUMNS_21))
        write_mode = st.radio("書き込み方法", ["置き換え（出走表の既存データを消して書き込む）", "追記"], horizontal=True, key="entry_write_mode")
        if st.button("📤 出走表タブへ書き込む", key="write_entry"):
            try:
                text = decode_bytes(up.getvalue()) if up is not None else raw_entry
                df_new = parse_input_table(text)
                sh = get_gspread_client()
                n = write_entry_buffer(sh, df_new, replace=write_mode.startswith("置き換え"))
                st.success(f"出走表タブへ {n} 行を書き込みました。下の「出走表を読み込む」を押してください。")
            except Exception as exc:
                st.error(f"出走表への書き込みに失敗しました: {exc}")

    if st.button("📥 出走表を読み込む", type="primary", key="load_entry", **STRETCH):
        try:
            sh = get_gspread_client()
            df_e, rids = load_entry_buffer(sh)
            if df_e.empty:
                st.warning("「出走表」タブに出走馬データがありません。上の入力欄から書き込むか、予想訂正で過去レースを読み戻してください。")
            else:
                set_entry(df_e, "buffer")
                st.success(f"{len(df_e)} 頭 / {len(rids)} レースを読み込みました（race_management に「{STATUS_UNPREDICTED}」で登録済み。Race ID の重複登録はありません）。")
        except Exception as exc:
            st.error(f"出走表の読み込みに失敗しました: {exc}")

    # --- 予想訂正（再読み込み） ---
    with st.expander("🔁 予想訂正（前回のデータを再読み込み）", expanded=False):
        st.caption("予想済みレースを Race ID で指定し、前回の出走データを読み戻して編集・再予測できます。再予測しても race_results / predictions / race_management の行は重複せず更新されます。")
        if st.button("📋 訂正できるレース一覧を読み込む", key="load_corr_list"):
            try:
                st.session_state["corr_registry_df"] = load_race_management_df(get_gspread_client(), fresh=True)
            except Exception as exc:
                st.error(f"レース一覧の読み込みに失敗しました: {exc}")
        reg_c = st.session_state.get("corr_registry_df")
        sel_rid = ""
        if reg_c is None:
            st.info("「訂正できるレース一覧を読み込む」を押してください。")
        else:
            pool = reg_c[reg_c["Status"].isin([STATUS_PREDICTED, STATUS_RESULT])]
            if pool.empty:
                st.info("訂正対象となる「予想済み」以降のレースがまだありません。")
            else:
                labels = {r["Race ID"]: f"{_race_label(r)}［{r['Status']}］" for _, r in pool.iterrows()}
                sel_rid = st.selectbox("訂正する Race ID", list(labels.keys()), format_func=lambda x: labels.get(x, x), key="corr_rid")
        manual_rid = st.text_input("Race ID を直接入力（任意・入力すると上の選択より優先）", key="corr_rid_manual")
        target_rid = manual_rid.strip() or sel_rid
        if target_rid:
            st.caption(f"対象 Race ID: `{target_rid}`")
        if st.button("🔁 予想訂正（前回のデータを再読み込み）", key="do_correction", disabled=not target_rid):
            try:
                df_c, note = load_race_for_correction(get_gspread_client(), target_rid)
                if df_c.empty:
                    st.error(note)
                else:
                    set_entry(df_c, "correction")
                    st.success(f"Race ID `{target_rid}` の {len(df_c)} 頭を読み戻しました。下の入力表で編集して再予測できます。")
                    if note:
                        st.warning(note)
            except Exception as exc:
                st.error(f"予想訂正データの読み込みに失敗しました: {exc}")

    # --- ② 編集 ---
    entry_df = st.session_state.get("entry_df")
    if entry_df is None or entry_df.empty:
        st.info("出走表はまだ読み込まれていません。「出走表を読み込む」または「予想訂正」を実行してください。")
    else:
        ver = st.session_state.get("entry_version", 0)
        work = entry_df.copy()
        work["_rid"] = work.apply(_race_id_from_row, axis=1)
        multi = work["_rid"].nunique() > 1
        src = {"buffer": "出走表タブ", "correction": "予想訂正（過去データ）"}.get(st.session_state.get("entry_source"), "手入力")
        first_meta = _race_meta_from_row(work.iloc[0])
        st.markdown(f"**② 出走馬の確認・編集**　（読み込み元: {src}）")
        if multi:
            st.warning(f"{work['_rid'].nunique()} レース分のデータが含まれています。レースごとに予想・保存します。")
        else:
            st.caption(f"{_race_label(first_meta)} ／ Race ID: `{work.iloc[0]['_rid'] or '（作成不可）'}`")
        st.caption("脚質は今回ではなく各馬の前走の脚質です（1=逃げ、2=先行、3=差し、4=追込）。分からない馬は空欄のままにしてください。")

        editor_df = pd.DataFrame({c: work[c].astype(str) for c in EDITOR_COLUMNS})
        editor_df["Tactics"] = [
            None if pd.isna(parse_tactics(x)) else str(int(parse_tactics(x))) for x in work["Tactics"]
        ]
        if multi:
            editor_df.insert(0, "Race ID", work["_rid"].values)
        col_cfg = {
            "Race ID": st.column_config.TextColumn("Race ID", disabled=True),
            "Horse Number": st.column_config.TextColumn("馬番"),
            "Horse Name": st.column_config.TextColumn("馬名"),
            "Bracket": st.column_config.TextColumn("枠"),
            "Sex": st.column_config.TextColumn("性"),
            "Horse Age": st.column_config.TextColumn("齢"),
            "Burden Weight": st.column_config.TextColumn("斤量"),
            "Jockey": st.column_config.TextColumn("騎手"),
            "Trainer": st.column_config.TextColumn("調教師"),
            "Tactics": st.column_config.SelectboxColumn("前走脚質", options=["1", "2", "3", "4"], required=False,
                                                         help="1=逃げ / 2=先行 / 3=差し / 4=追込"),
            "Last 3F": st.column_config.TextColumn("前走上がり3F"),
        }
        edited = st.data_editor(
            editor_df, hide_index=True, **STRETCH, num_rows="fixed",
            key=f"entry_editor_{ver}", column_config=col_cfg,
        )
        merged = entry_df.copy().reset_index(drop=True)
        for c in EDITOR_COLUMNS:
            merged[c] = [_blank(v) for v in edited[c].tolist()]

        if st.button("🔎 race_results から前走実績（脚質・上がり3F）を読み込む", key="load_previous_results"):
            try:
                sh = get_gspread_client()
                new, hit, cache = merged.copy(), 0, {}
                for i, row in new.iterrows():
                    day = row["Race Day"]
                    if day not in cache:
                        cache[day] = previous_result_lookup(sh, day)
                    r = cache[day].get(_norm_horse(row["Horse Name"]))
                    if r:
                        if pd.notna(r["Tactics"]):
                            new.at[i, "Tactics"] = str(int(r["Tactics"]))
                        if str(r["Last 3F"]).strip():
                            new.at[i, "Last 3F"] = str(r["Last 3F"])
                        hit += 1
                set_entry(new, st.session_state.get("entry_source", "buffer"), keep_results=True)
                flash("success", f"{hit} 頭の前走実績（脚質・上がり3F）を反映しました。")
                st.rerun()
            except Exception as exc:
                st.warning(f"前走実績の読み込みに失敗しました: {exc}")

        # 展開予想プレビュー（現在の入力）
        merged["_rid"] = merged.apply(_race_id_from_row, axis=1)
        for rid_p, g in merged.groupby("_rid", sort=False):
            p_live = diagnose_pace_from_codes(g["Tactics"].map(parse_tactics), len(g))
            render_pace(p_live, title=("展開予想プレビュー（現在の入力）" + (f"：{_race_label(_race_meta_from_row(g.iloc[0]))}" if multi else "")))

        run = st.button("🔮 AI予想を実行して保存", type="primary", **STRETCH, key="run_prediction")
        if run:
            try:
                sh = get_gspread_client()
                raw_all = merged.copy()
                raw_all["Horse Name"] = raw_all["Horse Name"].astype(str).str.strip()
                raw_all = raw_all[raw_all["Horse Name"] != ""].reset_index(drop=True)
                if raw_all.empty:
                    raise ValueError("馬名が入力された行がありません。")
                if (raw_all["_rid"] == "").any():
                    raise ValueError("Race Day / Venue / Race Name が空の行があり Race ID を作成できません。出走表を確認してください。")
                outputs = {}
                with st.spinner("AI予想を実行中…"):
                    for rid, g in raw_all.groupby("_rid", sort=False):
                        g = g.drop(columns=["_rid"]).reset_index(drop=True)
                        g["Previous_Tactics_Input"] = g["Tactics"].map(parse_tactics)
                        g["_in_tactics"] = g["Tactics"]
                        g["_in_last3f"] = g["Last 3F"]
                        out, view, pace = run_prediction_for_race(model, meta, g)
                        outputs[rid] = {"out": out, "view": view, "pace": pace, "label": _race_label(_race_meta_from_row(out.iloc[0]))}
                notice = []
                with st.spinner("race_results / race_management へ保存中…"):
                    for rid, o in outputs.items():
                        n = save_prediction_snapshot(sh, rid, o["out"], o["view"], o["pace"])
                        _how, status = upsert_race_management(sh, rid, _race_meta_from_row(o["out"].iloc[0]), STATUS_PREDICTED, len(o["out"]))
                        notice.append(f"`{rid}`：race_results に {n} 頭分を保存／ステータス「{status}」")
                st.session_state["results"] = outputs
                st.session_state["save_notice"] = notice
                try:
                    remain = clear_predicted_from_buffer(sh, list(outputs))
                    st.session_state["save_notice"].append(
                        "出走表タブをクリアしました（ヘッダーのみ）。" if remain == 0 else f"予想済みレースを出走表から削除しました（未予想の {remain} 行は残しています）。"
                    )
                except Exception as exc:
                    st.warning(f"予想の保存は完了しましたが、出走表タブのクリアに失敗しました: {exc}")
            except Exception as e:
                st.session_state.pop("results", None)
                st.error(f"予想または保存中にエラーが発生しました（出走表タブは消去していません）: {e}")
                with st.expander("詳細ログ"):
                    st.code(traceback.format_exc())

    # --- 予想結果の表示 ---
    results = st.session_state.get("results")
    if results:
        st.markdown("---")
        rid_list = list(results.keys())
        view_rid = rid_list[0]
        if len(rid_list) > 1:
            view_rid = st.selectbox("表示するレース", rid_list, format_func=lambda x: results[x]["label"], key="result_view_rid")
        R = results[view_rid]
        for line in st.session_state.get("save_notice", []):
            st.success(line)
        render_pace(R["pace"], title="③ 展開予想・脚質分布（ペース診断）")
        view_out = R["view"]
        st.subheader(f"④ AI予想結果（全{len(view_out)}頭・AI_Rank順）")
        st.caption(f"Race ID: `{view_rid}`　勝率は単勝モデル、複勝確率は3着以内モデルの予測値です。")
        view_mode = st.radio("表示スタイル切り替え", ["📱 スマホ用カード表示", "💻 従来テーブル表示"], horizontal=True, key="pred_view_mode")
        if view_mode == "📱 スマホ用カード表示":
            render_cards_prediction(view_out)
        else:
            show_cols = [c for c in view_out.columns if not c.startswith("_") and c not in ("TacticsCode", "Previous_Tactics_Input")]
            st.dataframe(view_out[show_cols], hide_index=True, **STRETCH, height=int(35 * (len(view_out) + 1) + 3))
        csv_download_button(R["out"], f"prediction_{view_rid}.csv", "📥 この予想をCSV保存（UTF-8 BOM付き）", key=f"dl_pred_{view_rid}")

# ------------------------------------------------------------
# TAB 2: AI予想 vs 実績（遅延ロード）＋ CSV出力
# ------------------------------------------------------------
with tab2:
    st.subheader("📊 AI予想 vs 実績 比較")
    st.caption("同一レースの「AIの予測順位・勝率」と「確定着順・上がり3F」を横並びで比較します。読み込みボタンを押したときだけデータを取得します。")
    if st.button("📥 データを読み込む", type="primary", key="load_tab2", **STRETCH):
        try:
            sh = get_gspread_client()
            st.session_state["t2_results"] = read_sheet_df(sh, RESULTS_SHEET, fresh=True)
            st.session_state["t2_pred"] = read_sheet_df(sh, PREDICTIONS_SHEET)
            st.session_state["t2_registry"] = load_race_management_df(sh, fresh=False)
            st.success("race_results / predictions / race_management を読み込みました。")
        except Exception as exc:
            st.error(f"データの読み込みに失敗しました: {exc}")

    t2_reg = st.session_state.get("t2_registry")
    t2_res = st.session_state.get("t2_results")
    t2_pred = st.session_state.get("t2_pred")
    if t2_reg is None:
        st.info("「データを読み込む」を押すと、比較したいレースを選択できます。")
    else:
        reg_sorted = t2_reg.copy()
        reg_sorted["_d"] = pd.to_datetime(reg_sorted["Race Day"], errors="coerce")
        reg_sorted = reg_sorted.sort_values("_d", ascending=False, na_position="last")
        labels2 = {r["Race ID"]: f"{_race_label(r)}［{r['Status']}］" for _, r in reg_sorted.iterrows()}
        # 管理シートに無いが race_results にある Race ID も選べるようにする
        if t2_res is not None and not t2_res.empty and "Race ID" in t2_res.columns:
            for rid_x in t2_res["Race ID"].astype(str).str.strip().unique():
                if rid_x and rid_x not in labels2:
                    labels2[rid_x] = f"{rid_x}［管理シート未登録］"
        if not labels2:
            st.info("比較できるレースがまだありません。")
        else:
            cmp_rid = st.selectbox("比較するレース", list(labels2.keys()), format_func=lambda x: labels2.get(x, x), key="t2_rid")
            cmp_df = build_comparison_from_results(t2_res if t2_res is not None else pd.DataFrame(), cmp_rid)
            if cmp_df.empty or (cmp_df["AI予想順位"].astype(str).str.strip() == "").all():
                alt = build_comparison_from_predictions(t2_pred, cmp_rid)
                if not alt.empty:
                    cmp_df = alt
                    st.caption("※ race_results に予想が無いため、predictions のデータから作成しています。")
            if cmp_df.empty:
                st.info("このレースには比較できるデータがありません（予想または結果が未登録です）。")
            else:
                no_result = cmp_df["確定着順"].astype(str).str.strip().eq("").all()
                no_ai = cmp_df["AI予想順位"].astype(str).str.strip().eq("").all()
                if no_result:
                    st.warning("確定着順がまだ登録されていません。「実績登録・一括登録」タブで登録すると比較できます。")
                if no_ai:
                    st.warning("このレースにはAI予想が保存されていません。")
                top = cmp_df[cmp_df["AI予想順位"].astype(str).str.strip() == "1"]
                m1, m2, m3 = st.columns(3)
                m1.metric("出走頭数", f"{len(cmp_df)}頭")
                m2.metric("AI1位の確定着順", (f"{top.iloc[0]['確定着順']}着" if not top.empty and str(top.iloc[0]['確定着順']).strip() else "-"))
                m3.metric("AI上位3頭のうち3着内", f"{int((cmp_df['AI上位3頭の的中'] == '○').sum())}頭")
                st.dataframe(cmp_df, hide_index=True, **STRETCH, height=int(35 * (len(cmp_df) + 1) + 3))
                csv_download_button(cmp_df, f"ai_vs_actual_{cmp_rid}_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv",
                                    "📥 CSV出力（Excel対応・UTF-8 BOM付き）", key="dl_compare")

    st.markdown("---")
    with st.expander("📄 predictions（学習用データ）を参照・絞り込み", expanded=False):
        if t2_pred is None:
            st.info("上の「データを読み込む」を押すと predictions が表示できます。")
        elif t2_pred.empty:
            st.info("predictions タブにデータがありません。")
        else:
            df_hist = ensure_21_columns(t2_pred.dropna(how="all"))
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                filter_race_name = st.text_input("レース名で検索", placeholder="例: アンタレス", key="t2_f_name")
            with fc2:
                filter_venue = st.multiselect("競馬場で絞り込み", options=sorted(v for v in df_hist["Venue"].unique() if v), key="t2_f_venue")
            with fc3:
                sort_col = st.selectbox("並び替えの基準列", options=list(df_hist.columns), index=0, key="t2_f_sort")
                ascending = st.radio("昇順/降順", options=["降順", "昇順"], index=0, key="t2_f_asc", horizontal=True) == "昇順"
            df_f = df_hist.copy()
            if filter_race_name:
                df_f = df_f[df_f["Race Name"].astype(str).str.contains(filter_race_name, na=False)]
            if filter_venue:
                df_f = df_f[df_f["Venue"].isin(filter_venue)]
            num_series = pd.to_numeric(df_f[sort_col], errors="coerce")
            if num_series.notna().sum() > 0:
                df_f = df_f.assign(_k=num_series).sort_values("_k", ascending=ascending).drop(columns=["_k"])
            else:
                df_f = df_f.sort_values(sort_col, ascending=ascending)
            df_f = df_f.reset_index(drop=True)
            st.success(f"表示中: **{len(df_f)} / {len(df_hist)} 件**")
            csv_download_button(df_f, f"predictions_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv", "📥 表示中のデータをCSV保存", key="dl_pred_hist")
            hist_mode = st.radio("表示スタイル切り替え", ["📱 スマホ用カード表示", "💻 従来テーブル表示"], horizontal=True, key="hist_view_mode")
            if hist_mode == "📱 スマホ用カード表示":
                render_cards_history(df_f)
            else:
                st.dataframe(df_f, **STRETCH, height=500)

# ------------------------------------------------------------
# TAB 3: 実績登録（race_results + predictions へ保存）／過去レース一括登録
# ------------------------------------------------------------
with tab3:
    st.subheader("🏁 レースを選択して結果を登録")
    st.caption(
        "実績（脚質・上がり3F・着順）を登録すると、①race_results のAI予想行へ実績を反映、"
        "②predictions へ「レース結果＋AI予想（勝率・予想順位のみ）」を保存、③race_management を「結果登録済み」に更新します。"
    )
    if st.button("📋 結果登録対象のレース一覧を読み込む", key="load_result_races"):
        try:
            st.session_state["t3_registry"] = load_race_management_df(get_gspread_client(), fresh=True)
            st.session_state.pop("t3_horses", None)
        except Exception as exc:
            st.error(f"レース一覧の読み込みに失敗しました: {exc}")

    t3_reg = st.session_state.get("t3_registry")
    if t3_reg is None:
        st.info("結果を登録するときに上のボタンを押してください（自動では読み込みません）。")
    elif t3_reg.empty:
        st.info("レース管理データがありません。先に出走表からAI予想を実行してください。")
    else:
        t3_sorted = t3_reg.copy()
        t3_sorted["_d"] = pd.to_datetime(t3_sorted["Race Day"], errors="coerce")
        t3_sorted = t3_sorted.sort_values("_d", ascending=False, na_position="last").reset_index(drop=True)
        t3_labels = {r["Race ID"]: f"{_race_label(r)}［{r['Status']}］" for _, r in t3_sorted.iterrows()}
        res_rid = st.selectbox("結果を登録するレース", list(t3_labels.keys()), format_func=lambda x: t3_labels.get(x, x), key="result_race_select")
        chosen = t3_sorted[t3_sorted["Race ID"] == res_rid].iloc[0]
        st.caption(f"Race ID: `{res_rid}` ／ 現在のステータス: **{chosen['Status'] or STATUS_UNPREDICTED}**")

        input_mode = st.radio("入力方法", ["📝 表で入力（予想時の馬リストを使う）", "📋 TSV / CSV を貼り付け"], horizontal=True, key="t3_input_mode")
        if input_mode.startswith("📝"):
            if st.button("🐴 このレースの馬リストを読み込む", key="load_t3_horses"):
                try:
                    res_df = read_sheet_df(get_gspread_client(), RESULTS_SHEET, fresh=True)
                    sub = res_df[res_df["Race ID"].astype(str).str.strip() == res_rid] if not res_df.empty and "Race ID" in res_df.columns else pd.DataFrame()
                    if sub.empty:
                        st.warning("このレースの馬リスト（race_results）がありません。TSV / CSV 貼り付けで登録してください。")
                    else:
                        for c in RESULT_REQUIRED:
                            if c not in sub.columns:
                                sub[c] = ""
                        horses = sub[RESULT_REQUIRED].astype(str).copy()
                        horses["_n"] = to_number(horses["Horse Number"]).fillna(999)
                        st.session_state["t3_horses"] = {
                            "rid": res_rid,
                            "df": horses.sort_values("_n", kind="stable").drop(columns=["_n"]).reset_index(drop=True),
                            "ver": st.session_state.get("t3_ver", 0) + 1,
                        }
                        st.session_state["t3_ver"] = st.session_state["t3_horses"]["ver"]
                except Exception as exc:
                    st.error(f"馬リストの読み込みに失敗しました: {exc}")
            hs = st.session_state.get("t3_horses")
            if hs and hs["rid"] == res_rid:
                st.caption("前走ではなく、このレースでの実際の脚質（1〜4）・上がり3F・確定着順を入力してください。")
                edited_res = st.data_editor(
                    hs["df"], hide_index=True, **STRETCH, num_rows="fixed", key=f"t3_editor_{res_rid}_{hs['ver']}",
                    column_config={
                        "Horse Number": st.column_config.TextColumn("馬番", disabled=True),
                        "Horse Name": st.column_config.TextColumn("馬名", disabled=True),
                        "Tactics": st.column_config.SelectboxColumn("実際の脚質", options=["1", "2", "3", "4"], required=False),
                        "Last 3F": st.column_config.TextColumn("上がり3F"),
                        "Placing": st.column_config.TextColumn("確定着順"),
                    },
                )
                if st.button("💾 選択したレースの実績を保存", type="primary", key="apply_results_table"):
                    try:
                        with st.spinner("保存中…"):
                            info = _do_register(get_gspread_client(), chosen, edited_res)
                        st.success(f"実績を保存しました。race_results 更新 {info['updated']} 行・新規 {info['appended']} 行 ／ predictions へ {info['train_rows']} 行を保存（置換 {info['train_replaced']} 行）／ ステータス「{STATUS_RESULT}」")
                        if info["matched_by_number"]:
                            st.warning("馬名が一致せず馬番で突き合わせた馬: " + "、".join(info["matched_by_number"]))
                        st.session_state.pop("t3_registry", None)
                    except Exception as exc:
                        st.error(f"実績の保存に失敗しました: {exc}")
                        with st.expander("詳細ログ"):
                            st.code(traceback.format_exc())
        else:
            result_text = st.text_area(
                "結果データ（TSV / CSV）", height=200, key="race_result_text",
                placeholder="Horse Number\tHorse Name\tTactics\tLast 3F\tPlacing\n1\tサンプルホースA\t2\t34.5\t1\n2\tサンプルホースB\t3\t35.1\t2",
            )
            if result_text.strip():
                try:
                    sep = "\t" if "\t" in result_text.strip().splitlines()[0] else ","
                    result_df = pd.read_csv(StringIO(result_text.strip().lstrip("\ufeff")), sep=sep, dtype=str, keep_default_na=False, engine="python")
                    result_df.columns = [RESULT_ALIASES.get(str(c).strip(), str(c).strip()) for c in result_df.columns]
                    if not all(c in result_df.columns for c in RESULT_REQUIRED):
                        st.error("ヘッダーは Horse Number / Horse Name / Tactics / Last 3F / Placing（または 馬番 / 馬名 / 脚質 / 上がり3F / 着順）にしてください。")
                    else:
                        result_df = result_df[RESULT_REQUIRED].fillna("").astype(str)
                        st.dataframe(result_df, **STRETCH)
                        if st.button("💾 選択したレースの実績を保存", type="primary", key="apply_results_text"):
                            try:
                                with st.spinner("保存中…"):
                                    info = _do_register(get_gspread_client(), chosen, result_df)
                                st.success(f"実績を保存しました。race_results 更新 {info['updated']} 行・新規 {info['appended']} 行 ／ predictions へ {info['train_rows']} 行を保存（置換 {info['train_replaced']} 行）／ ステータス「{STATUS_RESULT}」")
                                if info["matched_by_number"]:
                                    st.warning("馬名が一致せず馬番で突き合わせた馬: " + "、".join(info["matched_by_number"]))
                                st.session_state.pop("t3_registry", None)
                            except Exception as exc:
                                st.error(f"実績の保存に失敗しました: {exc}")
                                with st.expander("詳細ログ"):
                                    st.code(traceback.format_exc())
                except Exception as exc:
                    st.warning(f"結果データを解析できませんでした: {exc}")

    st.markdown("---")
    st.subheader("📥 過去レース結果データ 一括登録 ＆ エクスポート")
    st.caption("スプレッドシートのレース用タブを選択し、データの登録または全件CSVエクスポートが行えます（出走表・管理用シートは対象外）。")
    if st.button("📋 登録先タブ一覧を読み込む", key="load_import_tabs"):
        try:
            clear_sheet_cache()
            st.session_state["import_ws_list"] = [t for t in _cached_worksheet_titles(get_gspread_client().id) if t not in SYSTEM_SHEETS]
        except Exception as exc:
            st.error(f"タブ一覧の読み込みに失敗しました: {exc}")
    ws_list = st.session_state.get("import_ws_list", [])
    target_ws = None
    if ws_list:
        def_idx = ws_list.index(PREDICTIONS_SHEET) if PREDICTIONS_SHEET in ws_list else 0
        target_ws = st.selectbox("対象タブを選択", ws_list, index=def_idx, key="races_ws")
        if st.button("📥 選択したタブの全データを取得", key="fetch_target_ws"):
            try:
                st.session_state["target_export_df"] = read_sheet_df(get_gspread_client(), target_ws, fresh=True)
                st.session_state["target_export_ws"] = target_ws
            except Exception as exc:
                st.error(f"読み込みエラー: {exc}")
        if st.session_state.get("target_export_ws") == target_ws and st.session_state.get("target_export_df") is not None:
            df_all = st.session_state["target_export_df"]
            if df_all is not None and not df_all.empty:
                df_clean = ensure_21_columns(df_all.dropna(how="all").reset_index(drop=True))
                st.info(f"「{target_ws}」タブから全 **{len(df_clean)} 件** のデータを読み込みました。")
                csv_download_button(df_clean, f"{target_ws}_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv", f"📥 {target_ws} をCSV保存", key="dl_target_export")
                st.dataframe(df_clean, **STRETCH, height=200)
            else:
                st.info(f"「{target_ws}」タブにデータがありません。")
    else:
        st.info("スプレッドシートを読み込んでいません。「登録先タブ一覧を読み込む」を押してください。")

    st.markdown("**📝 過去レースデータのテキスト一括登録**")
    raw_races = st.text_area("過去レースデータの貼り付け（21列 TSV / CSV）", height=200, key="races_text", placeholder="\t".join(COLUMNS_21))
    if raw_races.strip():
        try:
            df_r = parse_input_table(raw_races)
            st.success(f"解析成功: **{len(df_r)} 行**（書き込み先: **{target_ws}**）")
            st.dataframe(df_r, **STRETCH)
            if target_ws and st.button(f"🚀 「{target_ws}」タブに21列データを一括書き込み", type="primary", key="bulk_write"):
                with st.spinner("書き込み中…"):
                    sh_w = get_gspread_client()
                    sh_w.worksheet(target_ws).append_rows(df_r.fillna("").astype(str).values.tolist(), value_input_option="USER_ENTERED", table_range="A1")
                    clear_sheet_cache()
                st.balloons()
                st.success(f"🎉 正常に {len(df_r)} 件のデータが追加されました！")
                if target_ws == PREDICTIONS_SHEET:
                    try:
                        added = backfill_management_from_predictions(sh_w)
                        st.info(f"race_management に新規 {added} レースを登録しました（既存の Race ID は重複登録しません）。")
                    except Exception as exc:
                        st.warning(f"race_management への登録に失敗しました。レース管理タブの取り込みボタンで再実行できます: {exc}")
        except Exception as e:
            st.error(f"データの解析または書き込みに失敗しました: {e}")

# ------------------------------------------------------------
# TAB 4: race_management（一元管理・フィルタ表示）
# ------------------------------------------------------------
with tab4:
    st.subheader("🗂️ レース管理（race_management）")
    st.caption("Race ID をユニークキーとして管理し、ステータスは「未予想 ➔ 予想済み ➔ 結果登録済み」と自動で進みます。")
    if st.button("🧾 空欄のタブに項目名（1行目）を自動入力", key="init_headers", **STRETCH):
        try:
            for title, msg in init_empty_headers(get_gspread_client()):
                st.write(f"・**{title}**：{msg}")
        except Exception as exc:
            st.error(f"項目名の入力に失敗しました: {exc}")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("📥 レース管理データを読み込む", type="primary", key="load_t4", **STRETCH):
            try:
                st.session_state["t4_registry"] = load_race_management_df(get_gspread_client(), fresh=True)
            except Exception as exc:
                st.error(f"レース管理データの読み込みに失敗しました: {exc}")
    with b2:
        if st.button("🔄 predictions から既存レースを取り込む", key="backfill_t4", **STRETCH):
            try:
                added = backfill_management_from_predictions(get_gspread_client())
                st.session_state["t4_registry"] = load_race_management_df(get_gspread_client(), fresh=True)
                st.success(f"新規 {added} レースを取り込みました（登録済みの Race ID は重複登録しません）。")
            except Exception as exc:
                st.error(f"取り込みに失敗しました: {exc}")

    t4 = st.session_state.get("t4_registry")
    if t4 is None:
        st.info("「レース管理データを読み込む」を押してください。")
    elif t4.empty:
        st.info("race_management にレースがまだ登録されていません。")
    else:
        c_a, c_b, c_c = st.columns(3)
        c_a.metric(STATUS_UNPREDICTED, f"{int((t4['Status'] == STATUS_UNPREDICTED).sum())}件")
        c_b.metric(STATUS_PREDICTED, f"{int((t4['Status'] == STATUS_PREDICTED).sum())}件")
        c_c.metric(STATUS_RESULT, f"{int((t4['Status'] == STATUS_RESULT).sum())}件")

        with st.expander("🔍 絞り込み（ステータス・日付・競馬場）", expanded=True):
            f1, f2 = st.columns(2)
            with f1:
                status_sel = st.multiselect("ステータス", list(STATUS_ORDER.keys()), default=list(STATUS_ORDER.keys()), key="t4_status")
            with f2:
                venue_sel = st.multiselect("競馬場", sorted(v for v in t4["Venue"].unique() if v), key="t4_venue")
            dates = pd.to_datetime(t4["Race Day"], errors="coerce")
            date_range = None
            if dates.notna().any():
                dmin, dmax = dates.min().date(), dates.max().date()
                date_range = st.date_input("日付範囲", value=(dmin, dmax), min_value=dmin, max_value=dmax, key="t4_dates")
            name_q = st.text_input("レース名で検索", key="t4_name")

        view4 = t4.copy()
        view4["_d"] = pd.to_datetime(view4["Race Day"], errors="coerce")
        if status_sel:
            view4 = view4[view4["Status"].isin(status_sel)]
        if venue_sel:
            view4 = view4[view4["Venue"].isin(venue_sel)]
        if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
            lo, hi = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            view4 = view4[view4["_d"].between(lo, hi) | view4["_d"].isna()]
        if name_q:
            view4 = view4[view4["Race Name"].astype(str).str.contains(name_q, na=False)]
        view4 = view4.sort_values("_d", ascending=False, na_position="last").drop(columns=["_d"]).reset_index(drop=True)
        st.success(f"表示中: **{len(view4)} / {len(t4)} 件**")
        st.dataframe(view4, hide_index=True, **STRETCH)
        csv_download_button(view4, f"race_management_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv", "📥 表示中の一覧をCSV保存（UTF-8 BOM付き）", key="dl_t4")

        with st.expander("🛠️ ステータスを手動で変更（訂正用）", expanded=False):
            if view4.empty:
                st.info("対象のレースがありません。")
            else:
                lab4 = {r["Race ID"]: f"{_race_label(r)}［{r['Status']}］" for _, r in view4.iterrows()}
                fix_rid = st.selectbox("対象レース", list(lab4.keys()), format_func=lambda x: lab4.get(x, x), key="t4_fix_rid")
                fix_status = st.selectbox("新しいステータス", list(STATUS_ORDER.keys()), key="t4_fix_status")
                if st.button("ステータスを更新", key="t4_fix_apply"):
                    try:
                        row4 = view4[view4["Race ID"] == fix_rid].iloc[0]
                        upsert_race_management(get_gspread_client(), fix_rid, _race_meta_from_row(row4), fix_status, force=True)
                        st.session_state["t4_registry"] = load_race_management_df(get_gspread_client(), fresh=True)
                        st.success("ステータスを更新しました。")
                    except Exception as exc:
                        st.error(f"ステータス更新に失敗しました: {exc}")

# ------------------------------------------------------------
# TAB 5: 血統データ登録 ＆ CSVダウンロード
# ------------------------------------------------------------
with tab5:
    st.subheader("🧬 血統データ管理（horses_master 専用）")
    st.caption("血統データ（馬名・父・母父）の登録および学習用CSVダウンロードが行えます。")

    with st.expander("📥 登録済みの血統データをCSVダウンロード", expanded=False):
        if st.button("📥 血統データを読み込む", key="load_horses_history"):
            try:
                st.session_state["horses_history_df"] = read_sheet_df(get_gspread_client(), HORSES_SHEET, fresh=True)
            except Exception as exc:
                st.warning(f"血統データの読み込みエラー: {exc}")
        df_horses_hist = st.session_state.get("horses_history_df")
        if df_horses_hist is not None:
            if not df_horses_hist.empty:
                df_horses_hist = df_horses_hist.dropna(how="all").reset_index(drop=True)
                for c in COLUMNS_HORSES:
                    if c not in df_horses_hist.columns:
                        df_horses_hist[c] = ""
                df_horses_hist = df_horses_hist[COLUMNS_HORSES].fillna("").astype(str)
                st.info(f"全 **{len(df_horses_hist)} 頭** の血統データを読み込みました。")
                csv_download_button(df_horses_hist, f"horses_master_export_{datetime.now(JST).strftime('%Y%m%d_%H%M%S')}.csv", "📥 horses_master をCSV保存", key="dl_horses")
                st.dataframe(df_horses_hist, **STRETCH, height=250)
            else:
                st.info("horses_master タブにデータがありません。")

    st.markdown("---")
    st.subheader("🧬 血統データ一括登録")
    raw_horses = st.text_area(
        "血統データの貼り付け（TSV / CSV）", height=220, key="horses_text",
        placeholder="Horse Name\tSire\tDam's Sire\nサンプルホース\tドゥラメンテ\tサンデーサイレンス",
    )
    if raw_horses.strip():
        try:
            sep = "\t" if "\t" in raw_horses else ","
            df_h = pd.read_csv(io.StringIO(raw_horses), sep=sep, dtype=str)
            for col in COLUMNS_HORSES:
                if col not in df_h.columns:
                    df_h[col] = ""
            df_h = df_h[COLUMNS_HORSES].fillna("").reset_index(drop=True).astype(str)
            st.success(f"解析成功: **{len(df_h)} 頭** の血統データ（書き込み先: **{HORSES_SHEET}**）")
            st.dataframe(df_h, **STRETCH)
            if st.button("🚀 horses_master タブに書き込む", type="primary", key="write_horses"):
                with st.spinner("書き込み中…"):
                    sh_h = get_gspread_client()
                    sh_h.worksheet(HORSES_SHEET).append_rows(df_h.values.tolist(), value_input_option="USER_ENTERED", table_range="A1")
                    clear_sheet_cache()
                st.balloons()
                st.success(f"🎉 正常に {len(df_h)} 頭の血統データが追加されました！")
        except Exception as e:
            st.error(f"データの解析または書き込みに失敗しました: {e}")
