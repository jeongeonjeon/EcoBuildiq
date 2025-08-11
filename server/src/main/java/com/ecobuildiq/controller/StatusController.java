package com.ecobuildiq.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;
import java.time.Instant;
import java.util.Map;

@RestController
public class StatusController {
    @GetMapping("/status")
    public Map<String, Object> status() {
        return Map.of("ok", true, "ts", Instant.now().toString());
    }
}
