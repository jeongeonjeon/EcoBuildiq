package com.ecobuildiq.service;

import com.ecobuildiq.model.ControlDecision;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class SafetyGuard {

    /** 간단 가드: 상호배제 + 히터 휴지시간 재확인 + all_off 지배 */
    public ControlDecision enforce(ControlDecision d, Map<String,Object> features, Map<String,Object> cfg) {
        if (d == null || d.getCommands() == null) return d;
        var cmd = d.getCommands();

        // 0) all_devices = OFF 가 들어오면 개별 장치 명령은 모두 OFF/정리 (야간 셧다운 등 지배)
        if ("OFF".equalsIgnoreCase(cmd.get("all_devices"))) {
            // 개별 장치 정리
            cmd.put("heater", "OFF");
            cmd.put("cooler", "OFF");
            cmd.put("fan", "OFF");
            cmd.put("light", "OFF");
            d.setRule(appendRule(d.getRule(), "R5_DOMINATES"));
            return d; // 이미 셧다운이니 이후 가드 불필요
        }

        // 1) heater/cooler 동시 ON 금지 → 간단히 cooler 제거(정책에서 더 정교하게 결정 가능)
        if ("ON".equalsIgnoreCase(cmd.get("heater")) && "ON".equalsIgnoreCase(cmd.get("cooler"))) {
            cmd.remove("cooler");
            d.setRule(appendRule(d.getRule(), "R7_MUTUAL_EXCLUSION"));
        }

        // 2) 휴지시간 위반 시 HOLD
        int lastOff = asInt(features.get("last_heater_off_minutes"));
        int restMin = asInt(cfg.getOrDefault("heater_rest_time", 30));
        if ("ON".equalsIgnoreCase(cmd.get("heater")) && lastOff < restMin) {
            cmd.put("heater", "HOLD");
            d.setRule(appendRule(d.getRule(), "R4_REST_GUARD"));
        }
        return d;
    }

    private static int asInt(Object o){ return o==null?0:Integer.parseInt(o.toString()); }

    private static String appendRule(String base, String tag) {
        if (base == null || base.isEmpty()) return tag;
        if (base.contains(tag)) return base;
        return base + "+" + tag;
    }
}
