package com.ecobuildiq.service;

import com.fasterxml.jackson.core.type.TypeReference;   // 
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.*;
import java.util.HashMap;
import java.util.Map;

@Service
public class ConfigService {
    private final Path path;
    private final ObjectMapper om = new ObjectMapper();

    public ConfigService() {
        this.path = Paths.get(System.getProperty("user.home"), ".ecobuildiq", "config.json");
        try { Files.createDirectories(this.path.getParent()); } catch (IOException ignored) {}
    }

    public Map<String,Object> get() {
        if (!Files.exists(path)) return defaults();
        try {
            // TypeReference로 정확한 제네릭 타입을 지정해 경고 제거
            return om.readValue(Files.readString(path), new TypeReference<Map<String,Object>>() {});
        } catch (Exception e) {
            return defaults();
        }
    }

    public Map<String,Object> save(Map<String,Object> cfg) {
        Map<String,Object> merged = new HashMap<>(defaults());
        if (cfg != null) merged.putAll(cfg);
        try { Files.writeString(path, om.writeValueAsString(merged)); } catch (IOException ignored) {}
        return merged;
    }

    private Map<String,Object> defaults() {
        return Map.of(
            "comfort_min_temp", 19,
            "comfort_max_temp", 24,
            "natural_light_threshold", 300,
            "heater_rest_time", 30,
            "active_start", "07:00",
            "active_end", "21:00"
        );
    }
}
