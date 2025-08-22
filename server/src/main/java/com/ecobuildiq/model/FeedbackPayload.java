package com.ecobuildiq.model;

public class FeedbackPayload {
    private String zone_id;
    private String timestamp;   // ISO8601 (옵션)
    private String signal;      // "too_cold" | "comfy" | "too_hot"

    public String getZone_id() { return zone_id; }
    public void setZone_id(String zone_id) { this.zone_id = zone_id; }
    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }
    public String getSignal() { return signal; }
    public void setSignal(String signal) { this.signal = signal; }
}
