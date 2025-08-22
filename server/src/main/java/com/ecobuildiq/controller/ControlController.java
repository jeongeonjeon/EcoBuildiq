package com.ecobuildiq.controller;

import com.ecobuildiq.model.ControlDecision;
import com.ecobuildiq.model.PolicyEvent;
import com.ecobuildiq.service.PolicyEvaluator;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

/**
 * 제어 평가 컨트롤러
 * - Python에서 전송한 PolicyEvent를 수신하여 평가 서비스로 전달
 * - 평탄화 필드와 기존 features 맵을 합성해 단일 맵으로 처리
 */
@RestController
@RequestMapping("/control") // ★ 최소 수정: base path 명시
public class ControlController {

    private final PolicyEvaluator evaluator;

    public ControlController(PolicyEvaluator evaluator) {
        this.evaluator = evaluator;
    }

    /**
     * 정책 평가 엔드포인트
     * - 요청 바디: PolicyEvent (snake_case 필드 포함)
     * - 응답: ControlDecision
     */
    @PostMapping("/evaluate")
    public ControlDecision evaluate(@RequestBody PolicyEvent event) {
        // 1) features 맵 복사
        Map<String, Object> fx = new HashMap<>();
        if (event.getFeatures() != null) {
            fx.putAll(event.getFeatures());
        }

        // 2) 평탄화 상위 필드도 규칙에서 쓰기 쉽게 합성
        if (event.getIndoorTemperaturePred() != null) {
            fx.put("indoor_temperature_pred", event.getIndoorTemperaturePred());
        }
        if (event.getOccupancyPred() != null) {
            fx.put("occupancy_pred", event.getOccupancyPred());
        }
        if (event.getHorizonMinutes() != null) {
            fx.put("horizon_minutes", event.getHorizonMinutes());
        }
        if (event.getCtxMeanLin() != null) {
            fx.put("ctx_mean_lin", event.getCtxMeanLin());
        }
        if (event.getCtxStdLin() != null) {
            fx.put("ctx_std_lin", event.getCtxStdLin());
        }
        if (event.getValue() != null) {
            fx.put("value", event.getValue());
        }
        if (event.getMeterType() != null) {
            fx.put("meter_type", event.getMeterType());
        }

        // 필요하면 meta도 evaluator로 넘기기 전에 합성 가능
        // if (event.getMeta() != null) fx.putAll(event.getMeta());

        // 3) 정책 평가 실행
        return evaluator.evaluate(event, fx);
    }
}
