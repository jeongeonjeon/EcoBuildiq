package com.ecobuildiq.model;

import com.fasterxml.jackson.annotation.JsonInclude;
import java.util.Map;

public class ControlDecision {
    private String rule;
    private Map<String, String> commands;

    // ← 추가: 설명(후보/점수 등)을 담는 필드. null이면 직렬화 생략
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Map<String, Object> explain;

    public ControlDecision() {
    }

    public ControlDecision(String rule, Map<String, String> commands) {
        this.rule = rule;
        this.commands = commands;
    }

    public String getRule() {
        return rule;
    }

    public void setRule(String rule) {
        this.rule = rule;
    }

    public Map<String, String> getCommands() {
        return commands;
    }

    public void setCommands(Map<String, String> commands) {
        this.commands = commands;
    }

    // ← 추가: getter/setter
    public Map<String, Object> getExplain() {
        return explain;
    }

    public void setExplain(Map<String, Object> explain) {
        this.explain = explain;
    }
}
