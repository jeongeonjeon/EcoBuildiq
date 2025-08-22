package com.ecobuildiq.controller;

import com.ecobuildiq.service.ConfigService;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/config")
public class ConfigController {
    private final ConfigService service;
    public ConfigController(ConfigService service) { this.service = service; }

    @GetMapping public Map<String,Object> get(){ return service.get(); }
    @PostMapping public Map<String,Object> save(@RequestBody Map<String,Object> cfg){ return service.save(cfg); }
}
