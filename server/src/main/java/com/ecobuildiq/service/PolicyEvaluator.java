// server/src/main/java/com/ecobuildiq/service/PolicyEvaluator.java
package com.ecobuildiq.service;

import com.ecobuildiq.model.ControlDecision;
import com.ecobuildiq.model.PolicyEvent;
import org.springframework.stereotype.Service;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.Map;

/**
 * PolicyEvaluator
 *
 * 목적:
 * - Python 추론(infer_lstm)의 결과(PolicyEvent)를 받아
 * - 사용자 설정(cfg: comfort_min_temp 등)을 적용하여
 * - 제어 명령(ControlDecision)을 산출한다.
 *
 * 규칙 카탈로그(우선순위 개념):
 * - R4 (safety): 히터 휴지시간 보호 → 가장 먼저 체크 (우선순위 최상)
 * - R1 (comfort): 점유=1 AND 온도예측 < comfort_min_temp → heater ON
 * - R2 (comfort): 점유=1 AND 온도예측 > comfort_max_temp → cooler ON
 * - R3 (economy): 점유=0 AND 조도>threshold → light OFF
 * - R5 (economy): 근무시간 외 AND 점유=0 → all_devices OFF
 * - NO_RULE: 위 어느 것도 아니면 현상 유지(또는 NO_ACTION)
 *
 * 주의:
 * - “안전 규칙”은 항상 최우선으로 평가해야 한다(예: R4).
 * - 동일 시각에 heater/cooler 동시 ON 같은 상호배제는 SafetyGuard에서 추가로 보강.
 * - 시간 비교는 문자열(HH:mm) 비교로 처리(간단/빠름). 경계가 복잡해지면 LocalTime 사용 고려.
 */
@Service
public class PolicyEvaluator {

    /**
     * @param e   PolicyEvent: infer에서 넘긴 이벤트 (features에 예측/상태 포함)
     * @param cfg 사용자 설정 맵: comfort_min_temp, active_start 등
     * @return ControlDecision: rule ID와 commands(예: {"heater":"ON"})
     */
    public ControlDecision evaluate(PolicyEvent e, Map<String, Object> cfg) {
        // 0) 입력/설정 파싱
        Map<String, Object> f = e.getFeatures() != null ? e.getFeatures() : Map.of();

        // infer가 만들어 준 주요 feature
        double tempPred = asDouble(f.get("temperature_pred")); // 온도 예측값
        int occupancy = asInt(f.get("occupancy")); // 0(무점유) / 1(점유)
        double lux = asDouble(f.getOrDefault("light_level", 0)); // 조도(lux), 없으면 0
        String heater = asString(f.get("heater_status"), "OFF"); // 현재 히터 상태
        int lastOffMin = asInt(f.get("last_heater_off_minutes")); // 마지막 off 후 경과분
        String hhmm = ensureHHmm(e); // "HH:mm" (features.time_hhmm 우선)

        // 사용자 설정(기본값 포함)
        double comfortMin = asDouble(cfg.getOrDefault("comfort_min_temp", 19)); // 난방 임계 하한
        double comfortMax = asDouble(cfg.getOrDefault("comfort_max_temp", 24)); // 냉방 임계 상한
        double lightTh = asDouble(cfg.getOrDefault("natural_light_threshold", 300));
        int restMin = asInt(cfg.getOrDefault("heater_rest_time", 30)); // 히터 휴지시간(분)
        String activeStart = asString(cfg.getOrDefault("active_start", "07:00"), "07:00");
        String activeEnd = asString(cfg.getOrDefault("active_end", "21:00"), "21:00");

        Map<String, String> cmd = new HashMap<>();

        // R4: 안전(SAFETY) - 히터 휴지시간 보호 (최우선)
        // - 현재 히터가 ON 상태이고, 마지막 OFF로부터 restMin분 미만이면 → HOLD
        // - 주: 이전 스니펫처럼 return 뒤에 두면 절대 실행되지 않음. 반드시 최상단에서 체크!
        if ("ON".equalsIgnoreCase(heater) && lastOffMin < restMin) {
            cmd.put("heater", "HOLD");
            return new ControlDecision("R4", cmd);
        }

        // R1: 쾌적성 - 점유 중 냉(추움) → 난방 ON
        // 조건: occupancy==1 AND tempPred < comfortMin
        if (occupancy == 1 && tempPred < comfortMin) {
            cmd.put("heater", "ON");
            return new ControlDecision("R1", cmd);
        }

        // R2: 쾌적성 - 점유 중 열(더움) → 냉방 ON
        // 조건: occupancy==1 AND tempPred > comfortMax
        if (occupancy == 1 && tempPred > comfortMax) {
            cmd.put("cooler", "ON");
            return new ControlDecision("R2", cmd);
        }

        // R3: 에너지 절약 - 무점유 & 자연광 충분 → 조명 OFF
        // 조건: occupancy==0 AND light_level > threshold
        if (occupancy == 0 && lux > lightTh) {
            cmd.put("light", "OFF");
            return new ControlDecision("R3", cmd);
        }

        // R5: 에너지 절약 - 근무시간 외 & 무점유 → 전부 OFF
        // 조건: time ∉ [activeStart, activeEnd] AND occupancy==0
        if (!within(hhmm, activeStart, activeEnd) && occupancy == 0) {
            cmd.put("all_devices", "OFF");
            return new ControlDecision("R5", cmd);
        }

        // 명시적 규칙이 없으면 NO_ACTION 또는 현상유지
        // 여기선 NO_ACTION 반환. 필요시 heater 상태 유지 등으로 바꿔도 됨.
        cmd.put("action", "NO_ACTION");
        return new ControlDecision("NO_RULE", cmd);
    }

    // 유틸: "HH:mm" 시간대 비교 (문자열 비교: 간단/빠름)
    // start <= hhmm <= end → true
    private static boolean within(String hhmm, String start, String end) {
        return hhmm.compareTo(start) >= 0 && hhmm.compareTo(end) <= 0;
    }

    // 유틸: 이벤트에서 HH:mm 확보
    // 1) features.time_hhmm 가 있으면 그대로 사용
    // 2) 없으면 timestamp(ISO8601)에서 HH:mm 추출
    // 3) 둘 다 실패 시 "00:00"
    private static String ensureHHmm(PolicyEvent e) {
        Object t = (e.getFeatures() != null ? e.getFeatures().get("time_hhmm") : null);
        if (t != null)
            return t.toString();
        try {
            OffsetDateTime odt = OffsetDateTime.parse(e.getTimestamp());
            return odt.toLocalTime().format(DateTimeFormatter.ofPattern("HH:mm"));
        } catch (Exception ex) {
            return "00:00";
        }
    }

    // 파싱 유틸: NPE/형변환 방지용
    private static double asDouble(Object o) {
        return o == null ? 0.0 : Double.parseDouble(o.toString());
    }

    private static int asInt(Object o) {
        return o == null ? 0 : Integer.parseInt(o.toString());
    }

    private static String asString(Object o, String d) {
        return o == null ? d : o.toString();
    }
}
