package com.ecobuildiq.model;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

/**
 * 정책 이벤트 입력 모델
 * - Python에서 보내는 평탄화(snake_case) 키를 @JsonProperty로 매핑
 * - building_id가 올 수도 있으므로 zone_id에 @JsonAlias("building_id")
 * - features/meta 맵은 그대로 유지(하위 호환)
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class PolicyEvent {

    /**
     * ISO-8601 문자열 권장 (예: 2025-08-14T09:00:00Z).
     * Jackson이 Instant로 바로 파싱 가능하지만, 문자열로 둬도 무방.
     */
    @JsonProperty("timestamp")
    private String timestamp;

    /** 핵심: Python이 building_id를 보내더라도 수용(별도 alias) */
    @JsonProperty("zone_id")
    @JsonAlias({ "building_id" })
    private String zone_id;

    @JsonProperty("meter_type")
    private String meterType;

    /** 예측 값 (kWh 등) */
    @JsonProperty("value")
    private Double value;

    /** 선택: 예측 맥락(선택) */
    @JsonProperty("indoor_temperature_pred")
    private Double indoorTemperaturePred;

    @JsonProperty("occupancy_pred")
    private Integer occupancyPred;

    /** 추론에서 넣는 경우가 있는 수평거리(분) */
    @JsonProperty("horizon_minutes")
    @JsonAlias({ "horizon" })
    private Double horizonMinutes;

    /** 선택: 표준화 컨텍스트(있는 경우만) */
    @JsonProperty("ctx_mean_lin")
    private Double ctxMeanLin;

    @JsonProperty("ctx_std_lin")
    private Double ctxStdLin;

    /** 기존 구조 유지: 자유형 feature/meta */
    @JsonProperty("features")
    private Map<String, Object> features;

    @JsonProperty("meta")
    private Map<String, Object> meta;

    // ----- getters / setters -----
    public String getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public String getZone_id() {
        return zone_id;
    }

    public void setZone_id(String zone_id) {
        this.zone_id = zone_id;
    }

    public String getMeterType() {
        return meterType;
    }

    public void setMeterType(String meterType) {
        this.meterType = meterType;
    }

    public Double getValue() {
        return value;
    }

    public void setValue(Double value) {
        this.value = value;
    }

    public Double getIndoorTemperaturePred() {
        return indoorTemperaturePred;
    }

    public void setIndoorTemperaturePred(Double indoorTemperaturePred) {
        this.indoorTemperaturePred = indoorTemperaturePred;
    }

    public Integer getOccupancyPred() {
        return occupancyPred;
    }

    public void setOccupancyPred(Integer occupancyPred) {
        this.occupancyPred = occupancyPred;
    }

    public Double getHorizonMinutes() {
        return horizonMinutes;
    }

    public void setHorizonMinutes(Double horizonMinutes) {
        this.horizonMinutes = horizonMinutes;
    }

    public Double getCtxMeanLin() {
        return ctxMeanLin;
    }

    public void setCtxMeanLin(Double ctxMeanLin) {
        this.ctxMeanLin = ctxMeanLin;
    }

    public Double getCtxStdLin() {
        return ctxStdLin;
    }

    public void setCtxStdLin(Double ctxStdLin) {
        this.ctxStdLin = ctxStdLin;
    }

    public Map<String, Object> getFeatures() {
        return features;
    }

    public void setFeatures(Map<String, Object> features) {
        this.features = features;
    }

    public Map<String, Object> getMeta() {
        return meta;
    }

    public void setMeta(Map<String, Object> meta) {
        this.meta = meta;
    }
}
