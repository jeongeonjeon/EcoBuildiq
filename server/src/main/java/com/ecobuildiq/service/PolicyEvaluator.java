package com.ecobuildiq.service;

import com.ecobuildiq.model.ControlDecision;
import com.ecobuildiq.model.PolicyEvent;
import org.springframework.stereotype.Service;

import java.time.OffsetDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import static java.lang.Math.*;

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
 * - R8 (comfort): 근무 시작 전 프리히트/프리쿨(리드타임 내) → heater/cooler ON
 * - R6 (IAQ): CO₂/RH 임계 초과 → fan ON
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
        double tempPred = asDouble(f.get("temperature_pred"));
        int occupancy = asInt(f.get("occupancy"));
        double lux = asDouble(f.getOrDefault("light_level", 0));
        double co2ppm = asDouble(f.getOrDefault("co2_ppm", 600));
        String heater = asString(f.get("heater_status"), "OFF");
        int lastOffMin = asInt(f.get("last_heater_off_minutes"));
        String hhmm = ensureHHmm(e);

        // 기존 cfg + (추가) 튜닝 가중치
        double comfortMin = asDouble(cfg.getOrDefault("comfort_min_temp", 19));
        double comfortMax = asDouble(cfg.getOrDefault("comfort_max_temp", 24));
        double lightTh = asDouble(cfg.getOrDefault("natural_light_threshold", 300));
        int restMin = asInt(cfg.getOrDefault("heater_rest_time", 30));
        String activeStart = asString(cfg.getOrDefault("active_start", "07:00"), "07:00");
        String activeEnd = asString(cfg.getOrDefault("active_end", "21:00"), "21:00");
        double co2Th = asDouble(cfg.getOrDefault("co2_threshold", 1000));
        double tempSigma = asDouble(cfg.getOrDefault("temp_sigma", 0.5));
        double priceKwh = asDouble(cfg.getOrDefault("energy_price", 0.20));

        // (추가) 가중치 슬라이더용 파라미터
        double W_COMFORT = asDouble(cfg.getOrDefault("w_comfort", 1.0));
        double W_COST = asDouble(cfg.getOrDefault("w_cost", 0.5));
        double W_UNCERT = asDouble(cfg.getOrDefault("w_uncert", 0.2));
        double W_SWITCH = asDouble(cfg.getOrDefault("w_switch", 0.3));
        double W_SCHED = asDouble(cfg.getOrDefault("w_sched", 0.4));

        // 후보/점수(설명용) 생성
        List<Map<String, Object>> candidates = buildCandidates(
                tempPred, occupancy, lux, co2ppm, heater, lastOffMin,
                comfortMin, comfortMax, lightTh, restMin, activeStart, activeEnd, co2Th, hhmm,
                tempSigma, priceKwh,
                W_COMFORT, W_COST, W_UNCERT, W_SWITCH, W_SCHED);

        Map<String, String> cmd = new HashMap<>();

        // R4: 안전(SAFETY) - 히터 휴지시간 보호 (최우선)
        // - 현재 히터가 ON 상태이고, 마지막 OFF로부터 restMin분 미만이면 → HOLD
        // - 주: 이전 스니펫처럼 return 뒤에 두면 절대 실행되지 않음. 반드시 최상단에서 체크!
        if ("ON".equalsIgnoreCase(heater) && lastOffMin < restMin) {
            cmd.put("heater", "HOLD");
            return decide("R4", cmd, candidates);
        }
        // R1: 쾌적성 - 점유 중 냉(추움) → 난방 ON
        // 조건: occupancy==1 AND tempPred < comfortMin
        if (occupancy == 1 && tempPred < comfortMin) {
            cmd.put("heater", "ON");
            return decide("R1", cmd, candidates);
        }
        // R2: 쾌적성 - 점유 중 열(더움) → 냉방 ON
        // 조건: occupancy==1 AND tempPred > comfortMax
        if (occupancy == 1 && tempPred > comfortMax) {
            cmd.put("cooler", "ON");
            return decide("R2", cmd, candidates);
        }
        
        if (co2ppm > co2Th) { // IAQ 데모 규칙
            cmd.put("fan", "ON");
            return decide("R6", cmd, candidates);
        }

        // R3: 에너지 절약 - 무점유 & 자연광 충분 → 조명 OFF
        // 조건: occupancy==0 AND light_level > threshold
        if (occupancy == 0 && lux > lightTh) {
            cmd.put("light", "OFF");
            return decide("R3", cmd, candidates);
        }

        // R5: 에너지 절약 - 근무시간 외 & 무점유 → 전부 OFF
        // 조건: time ∉ [activeStart, activeEnd] AND occupancy==0
        if (!within(hhmm, activeStart, activeEnd) && occupancy == 0) {
            cmd.put("all_devices", "OFF");
            return decide("R5", cmd, candidates);
        }

        // 명시적 규칙이 없으면 NO_ACTION 또는 현상유지
        // 여기선 NO_ACTION 반환. 필요시 heater 상태 유지 등으로 바꿔도 됨.
        cmd.put("action", "NO_ACTION");
        return decide("NO_RULE", cmd, candidates);
    }

    private List<Map<String, Object>> buildCandidates(
            double tempPred, int occupancy, double lux, double co2ppm, String heater, int lastOffMin,
            double cmin, double cmax, double lightTh, int restMin,
            String activeStart, String activeEnd, double co2Th, String hhmm,
            double sigma, double price,
            double W_COMFORT, double W_COST, double W_UNCERT, double W_SWITCH, double W_SCHED) {
        double P_HEATER = 1.5, P_COOLER = 1.2, P_FAN = 0.15, P_LIGHT = 0.10;
        boolean outOfHours = !within(hhmm, activeStart, activeEnd);
        List<Map<String, Object>> list = new ArrayList<>();
        java.util.function.BiFunction<String, Map<String, String>, Map<String, Object>> C = (id, commands) -> {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("commands", commands);
            return m;
        };

        // HEAT_ON
        {
            double comfortGain = (occupancy == 1 && tempPred < cmin) ? (cmin - tempPred) : 0.0;
            double energyKw = P_HEATER;
            double energyCost = energyKw * price;
            double switching = !"ON".equalsIgnoreCase(heater) ? 0.2 : 0.0;
            if (lastOffMin < restMin)
                switching += 10.0;
            double schedPen = (outOfHours && occupancy == 0) ? 0.5 : 0.0;

            double score = W_COMFORT * comfortGain - W_COST * energyCost - W_UNCERT * sigma
                    - W_SWITCH * switching - W_SCHED * schedPen;

            Map<String, Object> c = C.apply("HEAT_ON", Map.of("heater", "ON"));
            c.put("score", round2(score));
            c.put("why", Map.of("comfortGain", round2(comfortGain), "energyKw", round2(energyKw),
                    "energyCost", round2(energyCost), "uncertaintyStd", sigma,
                    "switchingPenalty", round2(switching)));
            list.add(c);
        }
        // COOL_ON
        {
            double comfortGain = (occupancy == 1 && tempPred > cmax) ? (tempPred - cmax) : 0.0;
            double energyKw = P_COOLER;
            double energyCost = energyKw * price;
            double switching = 0.2;
            double schedPen = (outOfHours && occupancy == 0) ? 0.5 : 0.0;

            double score = W_COMFORT * comfortGain - W_COST * energyCost - W_UNCERT * sigma
                    - W_SWITCH * switching - W_SCHED * schedPen;

            Map<String, Object> c = C.apply("COOL_ON", Map.of("cooler", "ON"));
            c.put("score", round2(score));
            c.put("why", Map.of("comfortGain", round2(comfortGain), "energyKw", round2(energyKw),
                    "energyCost", round2(energyCost), "uncertaintyStd", sigma,
                    "switchingPenalty", round2(switching)));
            list.add(c);
        }
        // FAN_ON
        {
            double comfortGain = (co2ppm > co2Th) ? Math.min((co2ppm - co2Th) / 400.0, 1.0) : 0.0;
            double energyKw = P_FAN;
            double energyCost = energyKw * price;
            double switching = 0.1;

            double score = W_COMFORT * comfortGain - W_COST * energyCost - W_UNCERT * sigma
                    - W_SWITCH * switching;

            Map<String, Object> c = C.apply("FAN_ON", Map.of("fan", "ON"));
            c.put("score", round2(score));
            c.put("why", Map.of("comfortGain", round2(comfortGain), "energyKw", round2(energyKw),
                    "energyCost", round2(energyCost), "uncertaintyStd", sigma,
                    "switchingPenalty", round2(switching)));
            list.add(c);
        }
        // LIGHT_OFF
        {
            double comfortGain = 0.0;
            double energyKw = (lux > lightTh) ? -P_LIGHT : 0.0; // 절감은 음수 kW
            double energyCost = energyKw * price;
            double switching = 0.05;

            double score = W_COMFORT * comfortGain - W_COST * energyCost - W_UNCERT * sigma
                    - W_SWITCH * switching;

            Map<String, Object> c = C.apply("LIGHT_OFF", Map.of("light", "OFF"));
            c.put("score", round2(score));
            c.put("why", Map.of("comfortGain", round2(comfortGain), "energyKw", round2(energyKw),
                    "energyCost", round2(energyCost), "uncertaintyStd", sigma,
                    "switchingPenalty", round2(switching)));
            list.add(c);
        }
        // NO_ACTION
        {
            double comfortGap = (tempPred < cmin) ? (cmin - tempPred) : (tempPred > cmax ? (tempPred - cmax) : 0.0);
            double comfortGain = (comfortGap == 0) ? 0.2 : -comfortGap;
            double energyKw = 0.0;

            double score = W_COMFORT * comfortGain - W_COST * 0.0 - W_UNCERT * sigma - W_SWITCH * 0.0;

            Map<String, Object> c = C.apply("NO_ACTION", Map.of("action", "NO_ACTION"));
            c.put("score", round2(score));
            c.put("why", Map.of("comfortGain", round2(comfortGain), "energyKw", round2(energyKw),
                    "energyCost", 0.0, "uncertaintyStd", sigma, "switchingPenalty", 0.0));
            list.add(c);
        }

        list.sort((a, b) -> Double.compare(asDouble(b.get("score")), asDouble(a.get("score"))));
        return list;
    }

    private ControlDecision decide(String rule, Map<String, String> cmd, List<Map<String, Object>> cands) {
        ControlDecision d = new ControlDecision(rule, cmd);
        Map<String, Object> ex = new HashMap<>();
        ex.put("candidates", cands);
        d.setExplain(ex);
        return d;
    }

    // 유틸(기존 유지)
    private static boolean within(String hhmm, String start, String end) {
        return hhmm.compareTo(start) >= 0 && hhmm.compareTo(end) <= 0;
    }

    // 유틸: 이벤트에서 HH:mm 확보
    // 1) features.time_hhmm 가 있으면 그대로 사용
    // 2) 없으면 timestamp(ISO8601)에서 HH:mm 추출
    // 3) 둘 다 실패 시 "00:00"
    private static String ensureHHmm(PolicyEvent e) {
        /* 동일 */ try {
            Object t = (e.getFeatures() != null ? e.getFeatures().get("time_hhmm") : null);
            if (t != null)
                return t.toString();
            OffsetDateTime odt = OffsetDateTime.parse(e.getTimestamp());
            return odt.toLocalTime().format(DateTimeFormatter.ofPattern("HH:mm"));
        } catch (Exception ex) {
            return "00:00";
        }
    }

    private static double asDouble(Object o) {
        try {
            return o == null ? 0.0 : Double.parseDouble(o.toString());
        } catch (Exception ex) {
            return 0.0;
        }
    }

    private static int asInt(Object o) {
        try {
            return o == null ? 0 : Integer.parseInt(o.toString());
        } catch (Exception ex) {
            return 0;
        }
    }

    private static String asString(Object o, String d) {
        return o == null ? d : o.toString();
    }

    private static double round2(double v) {
        return Math.round(v * 100.0) / 100.0;
    }
}
