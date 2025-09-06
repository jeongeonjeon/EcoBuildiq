# scripts/send_policy_events.py
import os
import csv
import json
import time
from datetime import datetime, timezone, timedelta
import requests

# 환경변수 기반 설정
JAVA_URL        = os.getenv("POLICY_API_URL", "http://localhost:8888/control/evaluate")  # 제어 API 엔드포인트
INPUT_CSV       = os.getenv("INPUT_CSV", "results/policy_events_ready.csv")              # 입력 CSV
OUT_JSONL       = os.getenv("OUT_JSONL", "results/policy_events.jsonl")                  # JSONL 출력
BATCH_SLEEP_SEC = float(os.getenv("BATCH_SLEEP_SEC", "0.02"))                             # 요청 간 대기(초)
DRY_RUN         = os.getenv("DRY_RUN", "0") == "1"                                       # 1이면 전송 생략
BASE_TS         = os.getenv("BASE_TS", "")                                               # preds_only용 기준 시각
METER_STATS     = os.getenv("METER_STATS", "results/meter_stats.json")                   # 선택적 통계 파일
MAX_ROWS        = int(os.getenv("MAX_ROWS", "0"))                                        # 0이면 무제한
LOG_EVERY       = int(os.getenv("LOG_EVERY", "0"))                                        # 0이면 로그 생략
TIMEZONE        = os.getenv("TIMEZONE", "UTC")

#Test
# 실험용 주입 옵션
FORCE_OCCUPANCY = os.getenv("FORCE_OCCUPANCY", "")  # "0" 또는 "1" (빈 문자열이면 미사용)
OCC_BY_TIME     = os.getenv("OCC_BY_TIME", "")      # 예: "07:00-21:00"
TEMP_CONST      = os.getenv("TEMP_CONST", "")       # 예: "20.5"

def occ_from_window(hhmm: str, win: str) -> int:
    try:
        s, e = win.split("-")
        return 1 if (s <= hhmm <= e) else 0
    except Exception:
        return 0
#####Test


# 선택적 meter 통계 로드 (없으면 무시)
_meter_stats = {}
try:
    if os.path.exists(METER_STATS):
        with open(METER_STATS, "r") as f:
            arr = json.load(f)
            for it in arr:
                m = str(it.get("meter") or it.get("meter_type") or "").strip()
                if not m:
                    continue
                _meter_stats[m] = {
                    "ctx_mean_lin": it.get("mean_lin"),
                    "ctx_std_lin": it.get("std_lin"),
                }
except Exception:
    _meter_stats = {}


# 유틸: 안전 파싱/폴백
def safe_float(x, default=0.0):
    try:
        if x is None or x == "":
            return default
        return float(str(x).strip())
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        if x is None or x == "":
            return default
        s = str(x).strip().lower()
        if s in ("true", "yes"): return 1
        if s in ("false", "no"): return 0
        return int(float(s))  # "1.0" 같은 문자열도 허용
    except Exception:
        return default

def norm_on_off(x, default="OFF"):
    s = str(x).strip().upper() if x is not None else default
    if s in ("1", "ON", "TRUE", "YES"): return "ON"
    if s in ("0", "OFF", "FALSE", "NO"): return "OFF"
    return default

def hhmm_from_row(row):
    # time_hhmm 컬럼이 있으면 그대로, 없으면 timestamp/ts_iso에서 추출
    if "time_hhmm" in row and row["time_hhmm"]:
        s = str(row["time_hhmm"])
        return s if ":" in s else f"{s[:2]}:{s[2:]}"  # 0905 -> 09:05
    ts = row.get("timestamp") or row.get("ts_iso") or ""
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return f"{dt.hour:02d}:{dt.minute:02d}"       # ← "HH:MM"
    except Exception:
        return "00:00"


def parse_base_ts(base_ts: str):
    if not base_ts:
        return None
    try:
        # 예: "2025-08-14T00:00:00Z"
        if base_ts.endswith("Z"):
            return datetime.fromisoformat(base_ts.replace("Z", "+00:00")).astimezone(timezone.utc)
        return datetime.fromisoformat(base_ts).astimezone(timezone.utc)
    except Exception:
        return None

def tz_aware_now():
    try:
        # TIMEZONE 미사용 시 UTC
        return datetime.now(timezone.utc)
    except Exception:
        return datetime.utcnow().replace(tzinfo=timezone.utc)

# 스키마 판별
def guess_schema(headers):
    """
    - policy_ready: infer_lstm가 만든 평탄화 CSV
      기존: ["timestamp","zone_id","meter_type","value"]
      # === PATCH: building_id alias 허용 ===
      새로: ["timestamp","building_id","meter_type","value"]도 인식
    - preds_only: ["meter_type","seq_index","y_pred"]
    """
    h = set([c.strip().lower() for c in headers])
    if {"timestamp", "zone_id", "meter_type", "value"}.issubset(h):
        return "policy_ready"
    # === PATCH: building_id일 때도 policy_ready로 처리
    if {"timestamp", "building_id", "meter_type", "value"}.issubset(h):
        return "policy_ready"
    if {"meter_type", "seq_index", "y_pred"}.issubset(h):
        return "preds_only"
    return "unknown"


# 평탄화 → features/meta 변환
def to_features_event(flat: dict) -> dict:
    """
    flat: {"timestamp","zone_id","meter_type","value", ...옵션필드}
    반환: {"timestamp","zone_id","features":{...},"meta":{...}}
    """
    ts = str(flat.get("timestamp") or "")
    zone = str(flat.get("zone_id") or "default")
    meter = str(flat.get("meter_type") or "default")

    features = {
        "temperature_pred": safe_float(
            flat.get("indoor_temperature_pred") or flat.get("temperature_pred") or flat.get("temp_pred"), 0.0
        ),
        "occupancy": safe_int(flat.get("occupancy") or flat.get("occupancy_pred"), 0),
        "light_level": safe_float(flat.get("light_level"), 0.0),
        "time_hhmm": flat.get("time_hhmm") or hhmm_from_row(flat),
        "heater_status": norm_on_off(flat.get("heater_status"), "OFF"),
        "last_heater_off_minutes": safe_int(
            flat.get("last_heater_off_minutes") or flat.get("last_off_min"), 9999
        ),
        "value": safe_float(flat.get("value"), 0.0),
        "horizon_minutes": safe_float(flat.get("horizon_minutes"), 0.0),
    }

    meta = {
        "meter_type": meter,
        "run_id": flat.get("run_id") or "",
        # 필요하면 building_id 원본을 meta에 남길 수도 있음:
        # "building_id_raw": flat.get("building_id"),
    }
#####Test
    # --- occupancy 주입 ---
    if FORCE_OCCUPANCY != "":
        features["occupancy"] = safe_int(FORCE_OCCUPANCY, features.get("occupancy", 0))
    elif OCC_BY_TIME:
        features["occupancy"] = occ_from_window(features.get("time_hhmm", "00:00"), OCC_BY_TIME)

    # --- temperature_pred 주입 ---
    if (features.get("temperature_pred", 0.0) == 0.0) and TEMP_CONST not in ("", None):
        features["temperature_pred"] = safe_float(TEMP_CONST, 0.0)
#####Test

    return {
        "timestamp": ts,
        "zone_id": zone,
        "features": features,
        "meta": meta,
    }

# CSV → 평탄화 이벤트
def row_to_event_policy_ready(row):
    """
    기존 평탄화 포맷. zone_id 기반.
    # === PATCH: building_id alias ===
    zone_id가 없으면 building_id를 zone_id로 사용.
    """
    # === PATCH: building_id alias ===
    zone_alias = row.get("zone_id") or row.get("building_id") or "default"
    meter_alias = (row.get("meter_type") or row.get("meter")
                    or row.get("type") or row.get("metric"))
    meter_val = str(meter_alias) if meter_alias not in (None, "") else "default"

    event = {
        "timestamp": str(row.get("timestamp")),
        "zone_id": str(zone_alias),
        "meter_type": meter_val,            # ← 변경 반영
        "value": safe_float(row.get("value"), 0.0),
    }

    if "indoor_temperature_pred" in row:
        event["indoor_temperature_pred"] = safe_float(row.get("indoor_temperature_pred"), 0.0)
    if "occupancy_pred" in row:
        event["occupancy_pred"] = safe_int(row.get("occupancy_pred"), 0)
    if "light_level" in row:
        event["light_level"] = safe_float(row.get("light_level"), 0.0)
    if "heater_status" in row:
        event["heater_status"] = row.get("heater_status")
    if "last_heater_off_minutes" in row:
        event["last_heater_off_minutes"] = safe_int(row.get("last_heater_off_minutes"), 9999)
    if "horizon_minutes" in row:
        event["horizon_minutes"] = safe_float(row.get("horizon_minutes"), 0.0)
    elif "horizon" in row:
        event["horizon_minutes"] = safe_float(row.get("horizon"), 0.0)

    # 선택적 meter 통계 추가 (ctx_ 접두어)
    st = _meter_stats.get(str(row.get("meter_type")), {})
    event.update(st)
    return event

def row_to_event_preds_only(row, base_ts_utc):
    """
    최소 예측 포맷:
      ["meter_type","seq_index","y_pred"] (+optional y_true)
    BASE_TS가 있으면 seq_index * 1분 가산하여 timestamp 계산.
    """
    meter_val = row.get("meter_type")
    seq = int(float(row.get("seq_index"))) if row.get("seq_index") not in (None, "") else 0
    y_pred = safe_float(row.get("y_pred"), 0.0)

    if base_ts_utc is not None:
        ts_dt = base_ts_utc + timedelta(minutes=seq)
    else:
        # BASE_TS 미설정 시, 현재 시각 사용
        ts_dt = tz_aware_now()

    ts = ts_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    event = {
        "timestamp": ts,
        "zone_id": "default",  # preds_only는 대상 zone 정보가 없으므로 default
        "meter_type": str(meter_val),
        "value": y_pred,
        "horizon_minutes": seq,
    }
    if "y_true" in row:
        event["y_true"] = safe_float(row.get("y_true"), 0.0)
    return event


# 전송
def post_event(ev: dict):
    """
    ev: {"timestamp","zone_id","features":{...},"meta":{...}}
    """
    if DRY_RUN:
        # 드라이런이면 네트워크 생략, 대신 OK로 간주
        return True, {"dry_run": True}
    try:
        r = requests.post(JAVA_URL, json=ev, timeout=10)
        ok = (200 <= r.status_code < 300)
        return ok, {"status_code": r.status_code, "text": r.text[:200]}
    except Exception as e:
        return False, {"error": str(e)}

# 메인
def main():
    ok = skipped = failed = 0

    with open(INPUT_CSV, "r") as f_in, open(OUT_JSONL, "w") as f_out:
        reader = csv.DictReader(f_in)
        headers = reader.fieldnames or []
        schema = guess_schema(headers)
        base_ts_utc = parse_base_ts(BASE_TS)

        print("Detected schema:", schema)
        if schema == "preds_only" and base_ts_utc is None:
            print("BASE_TS is not set. Timestamps will use current time baseline.")

        for i, row in enumerate(reader, start=1):
            if MAX_ROWS and i > MAX_ROWS:
                break

            # 1) CSV → 평탄화 이벤트
            if schema == "policy_ready":
                flat = row_to_event_policy_ready(row)            # (zone_id) or (building_id→zone_id alias)
            elif schema == "preds_only":
                flat = row_to_event_preds_only(row, base_ts_utc)
            else:
                skipped += 1
                continue

            # 2) 유효성(최소) 체크
            if not flat.get("timestamp") or not flat.get("zone_id") or not flat.get("meter_type"):
                skipped += 1
                continue

            # 3) 평탄화 → features/meta
            ev = to_features_event(flat)

            # 4) 전송 + JSONL 기록
            ok_flag, info = post_event(ev)
            if ok_flag:
                ok += 1
            else:
                failed += 1

            # JSONL로 항상 기록(재현/리플레이 용)
            f_out.write(json.dumps({"request": ev, "result": info}, ensure_ascii=False) + "\n")

            if LOG_EVERY and (i % LOG_EVERY == 0):
                print(f"Progress: {i} rows (OK={ok}, Failed={failed}, Skipped={skipped})", flush=True)

            time.sleep(BATCH_SLEEP_SEC)

    if DRY_RUN:
        print("Dry run completed.")
    print("Sent OK:", ok, "Skipped:", skipped, "Failed:", failed)
    print("JSONL saved to", OUT_JSONL)


if __name__ == "__main__":
    main()
