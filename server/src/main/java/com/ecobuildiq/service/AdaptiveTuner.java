package com.ecobuildiq.service;

import com.ecobuildiq.model.FeedbackPayload;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * AdaptiveTuner
 *
 * 목적:
 *  - 사용자 피드백(too_cold/comfy/too_hot)을 받아 zone별 온열 임계값을 미세 조정(오프셋)한다.
 *  - "베이스 설정" + "zone별 오프셋" = "실효 임계값" (PolicyEvaluator에 적용)
 *
 * 규칙(간단한 온라인 학습):
 *  - too_cold  : comfort_min_temp_z += step
 *  - too_hot   : comfort_max_temp_z -= step
 *  - comfy     : 오프셋을 0쪽으로 α-감쇠(과적합 방지)
 * 안전장치:
 *  - deadband(최소 간격) 유지: comfort_max >= comfort_min + deadband
 *  - global bounds: [minBound, maxBound] 범위 고정
 *  - TTL/감쇠: 시간이 지나면 오프셋이 서서히 줄어듦 (여기선 comfy로 근사)
 */
@Service
public class AdaptiveTuner {

    private final ConfigService config; // 베이스 설정
    private final ObjectMapper om = new ObjectMapper();

    // zone별 오프셋 저장: { zone -> { "d_min": +0.4, "d_max": -0.2 } }
    private final Map<String, Map<String, Double>> zoneOffsets = new ConcurrentHashMap<>();

    // 영속화 위치: ~/.ecobuildiq/offsets.json
    private final Path offsetsPath;

    // 하이퍼파라미터(필요 시 /config에 노출 가능)
    private final double step = 0.2;           // 한 번 피드백에 조정되는 섭씨
    private final double comfyDecay = 0.5;     // comfy 시, offset을 절반으로 감쇠
    private final double deadband = 1.0;       // 최소 온열 간격(°C)
    private final double minBound = 16.0;      // 전체 최소(과제/안전용)
    private final double maxBound = 28.0;      // 전체 최대

    public AdaptiveTuner(ConfigService config) {
        this.config = config;
        this.offsetsPath = Paths.get(System.getProperty("user.home"), ".ecobuildiq", "offsets.json");
        loadOffsets();
    }

    /** 피드백 적용 후, 해당 zone의 "실효 임계값"을 반환 */
    public Map<String,Object> applyFeedback(FeedbackPayload fb) {
        String zone = fb.getZone_id();
        if (zone == null || zone.isBlank()) zone = "default";

        var base = config.get(); // 베이스 설정
        var offs = zoneOffsets.computeIfAbsent(zone, z -> new HashMap<>(Map.of("d_min",0.0, "d_max",0.0)));

        double dmin = offs.getOrDefault("d_min", 0.0);
        double dmax = offs.getOrDefault("d_max", 0.0);

        switch (String.valueOf(fb.getSignal())) {
            case "too_cold":
                dmin += step; // 더 따뜻하게
                break;
            case "too_hot":
                dmax -= step; // 더 시원하게
                break;
            case "comfy":
                // 오프셋을 원점으로 수축 → 과도한 편향 누적 방지
                dmin *= comfyDecay;
                dmax *= comfyDecay;
                break;
            default:
                // no-op
        }

        // deadband 및 global bounds 보정
        double baseMin = asDouble(base.getOrDefault("comfort_min_temp", 19));
        double baseMax = asDouble(base.getOrDefault("comfort_max_temp", 24));
        double effMin = clamp(baseMin + dmin, minBound, maxBound - deadband);
        double effMax = clamp(baseMax + dmax, effMin + deadband, maxBound);

        // 보정 후 다시 오프셋 저장(베이스 기준)
        offs.put("d_min", effMin - baseMin);
        offs.put("d_max", effMax - baseMax);
        zoneOffsets.put(zone, offs);
        persistOffsets();

        return Map.of(
            "zone", zone,
            "comfort_min_temp", effMin,
            "comfort_max_temp", effMax,
            "offsets", offs
        );
    }

    /** PolicyEvaluator용: zone별 실효 임계값(베이스+오프셋) */
    public Map<String,Object> getEffectiveForZone(String zoneId) {
        var base = new HashMap<>(config.get());
        var offs = zoneOffsets.getOrDefault(zoneId, Map.of("d_min",0.0, "d_max",0.0));

        double baseMin = asDouble(base.getOrDefault("comfort_min_temp", 19));
        double baseMax = asDouble(base.getOrDefault("comfort_max_temp", 24));
        double effMin = clamp(baseMin + offs.getOrDefault("d_min",0.0), minBound, maxBound - deadband);
        double effMax = clamp(baseMax + offs.getOrDefault("d_max",0.0), effMin + deadband, maxBound);

        base.put("comfort_min_temp", effMin);
        base.put("comfort_max_temp", effMax);
        base.put("offsets", offs);
        base.put("deadband", deadband);
        return base;
    }

    // 조건 persistence 조건
    private void loadOffsets() {
        try {
            if (Files.exists(offsetsPath)) {
                var txt = Files.readString(offsetsPath);
                // 원본: var map = om.readValue(txt, Map.class);  // unchecked conversion 경고
                var raw = om.readValue(txt, Map.class);
                var casted = safeCastToStringDoubleMap(raw);     // ✅ 안전 변환
                zoneOffsets.clear();
                zoneOffsets.putAll(casted);
            } else {
                Files.createDirectories(offsetsPath.getParent());
            }
        } catch (Exception ignored) {}
    }
    private void persistOffsets() {
        try {
            Files.writeString(offsetsPath, om.writeValueAsString(zoneOffsets));
        } catch (IOException ignored) {}
    }

    // 조건 utils 조건
    private static double asDouble(Object o){ return o==null?0.0:Double.parseDouble(o.toString()); }
    private static double clamp(double v, double lo, double hi){ return Math.max(lo, Math.min(hi, v)); }

    // ===== 최소 패치: 안전 캐스팅 헬퍼 (unchecked 경고 제거) =====
    @SuppressWarnings("unchecked")
    private static Map<String, Map<String, Double>> safeCastToStringDoubleMap(Object in) {
        Map<String, Map<String, Double>> out = new HashMap<>();
        if (!(in instanceof Map<?, ?> rawOuter)) return out;

        for (Map.Entry<?, ?> e : rawOuter.entrySet()) {
            String outerKey = String.valueOf(e.getKey());
            Object inner = e.getValue();
            if (inner instanceof Map<?, ?> rawInner) {
                Map<String, Double> innerMap = new HashMap<>();
                for (Map.Entry<?, ?> ie : rawInner.entrySet()) {
                    String innerKey = String.valueOf(ie.getKey());
                    Object val = ie.getValue();
                    if (val instanceof Number n) {
                        innerMap.put(innerKey, n.doubleValue());
                    } else if (val != null) {
                        try { innerMap.put(innerKey, Double.parseDouble(val.toString())); }
                        catch (NumberFormatException ignore) {}
                    }
                }
                out.put(outerKey, innerMap);
            }
        }
        return out;
    }
}
