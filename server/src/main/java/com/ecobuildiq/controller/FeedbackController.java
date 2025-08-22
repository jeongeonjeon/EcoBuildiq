package com.ecobuildiq.controller;

import com.ecobuildiq.model.FeedbackPayload;
import com.ecobuildiq.service.AdaptiveTuner;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/feedback")
public class FeedbackController {

    private final AdaptiveTuner tuner;

    public FeedbackController(AdaptiveTuner tuner) {
        this.tuner = tuner;
    }

    /**
     * 사용자 피드백 수신 (too_cold / comfy / too_hot)
     * body 예:
     * {
     *   "zone_id":"office_101",
     *   "timestamp":"2025-08-11T10:05:00Z",
     *   "signal":"too_cold"
     * }
     */
    @PostMapping
    public Map<String,Object> submit(@RequestBody FeedbackPayload fb) {
        var updated = tuner.applyFeedback(fb); // zone별 임계값 오프셋 갱신
        return Map.of(
            "ok", true,
            "zone_id", fb.getZone_id(),
            "effective_config", updated // 갱신된 실제 임계값(베이스+오프셋)
        );
    }

    /** 현재 zone별 실효 임계값(베이스+오프셋) 조회 */
    @GetMapping("/effective/{zone}")
    public Map<String,Object> effective(@PathVariable("zone") String zoneId) {
        return tuner.getEffectiveForZone(zoneId);
    }
}
