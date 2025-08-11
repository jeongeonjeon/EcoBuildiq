package com.ecobuildiq.service;

import com.ecobuildiq.model.ControlDecision;
import com.ecobuildiq.model.PolicyEvent;
import java.util.HashMap;
import java.util.Map;

public class PolicyEvaluator {
    public ControlDecision evaluate(PolicyEvent e, Map<String, Object> cfg) {
        Map<String, Object> f = e.getFeatures();

        double t = asDouble(f.get("temperature_pred"));
        boolean occ = asInt(f.get("occupancy")) == 1;
        String heaterStatus = asString(f.get("heater_status"), "OFF");
        int lastOff = asInt(f.get("last_heater_off_minutes"));

        int comfortMin = asInt(cfg.getOrDefault("comfort_min_temp", 19));
        int heaterRest = asInt(cfg.getOrDefault("heater_rest_time", 30));

        // R4 휴지시간 보호
        if ("ON".equals(heaterStatus) && lastOff < heaterRest) {
            return new ControlDecision("R4", Map.of("heater", "HOLD"));
        }
        // R1 난방 온
        if (t < comfortMin && occ) {
            return new ControlDecision("R1", Map.of("heater", "ON"));
        }
        return new ControlDecision("NO_RULE", Map.of("action", "NO_ACTION"));
    }

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
