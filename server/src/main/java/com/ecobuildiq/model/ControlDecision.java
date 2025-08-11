package com.ecobuildiq.model;

import java.util.Map;

public class ControlDecision {
    private String rule;
    private Map<String, String> commands;

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
}
