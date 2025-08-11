package com.ecobuildiq.controller;

import com.ecobuildiq.model.PolicyEvent;
import com.ecobuildiq.service.PolicyEvaluator;
import com.ecobuildiq.model.ControlDecision;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/control")
public class ControlController {

    private PolicyEvaluator evaluator;

    @PostMapping("/evaluate")
    public ControlDecision evaluate(@RequestBody PolicyEvent event) {
        Map<String, Object> cfg = Map.of(
                "comfort_min_temp", 19,
                "heater_rest_time", 30);
        return evaluator.evaluate(event, cfg);

        // // 이후 PolicyEvaluator에 위임할 예정. 지금은 더미 결과 반환
        // return new ControlDecision("NO_RULE", Map.of("action", "NO_ACTION"));
    }
}
