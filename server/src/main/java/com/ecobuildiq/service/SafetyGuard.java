package com.ecobuildiq.service;

import com.ecobuildiq.model.ControlDecision;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class SafetyGuard {

    /** 간단 가드: 상호배제 + 히터 휴지시간 재확인 */
    public ControlDecision enforce(ControlDecision d, Map<String,Object> features, Map<String,Object> cfg) {
        if (d == null || d.getCommands() == null) return d;
        var cmd = d.getCommands();

        // 1) heater/cooler 동시 ON 금지 → heater 우선 (필요시 정책으로 결정)
        if ("ON".equalsIgnoreCase(cmd.get("heater")) && "ON".equalsIgnoreCase(cmd.get("cooler"))) {
            cmd.remove("cooler");
            d.setRule("R7_MUTUAL_EXCLUSION");
        }

        // 2) 휴지시간 위반 시 HOLD
        int lastOff = asInt(features.get("last_heater_off_minutes"));
        int restMin = asInt(cfg.getOrDefault("heater_rest_time", 30));
        if ("ON".equalsIgnoreCase(cmd.get("heater")) && lastOff < restMin) {
            cmd.put("heater", "HOLD");
            d.setRule("R4_REST_GUARD");
        }
        return d;
    }

    private static int asInt(Object o){ return o==null?0:Integer.parseInt(o.toString()); }
}
