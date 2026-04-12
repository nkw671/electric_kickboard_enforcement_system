package com.kickboard.back.dto;

import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.kickboard.back.entity.ViolationRecord;
import lombok.Getter;
import java.time.LocalDateTime;

// 프론트엔드로 단속 기록을 전달할 때 사용하는 응답용 DTO
@Getter
public class ViolationResponse {

    private Long id;

    @JsonProperty("type")
    private String violationType;

    @JsonProperty("image_url")
    private String imageUrl;

    @JsonProperty("timestamp")
    @JsonFormat(pattern = "yyyy-MM-dd HH:mm:ss", timezone = "Asia/Seoul")
    private LocalDateTime createdAt;

    private String camera;

    private Integer confidence;

    public ViolationResponse(ViolationRecord record) {
        this.id = record.getId();
        this.violationType = record.getViolationType();
        this.imageUrl = record.getImageUrl();
        this.createdAt = record.getCreatedAt();
        this.camera = record.getCamera();
        this.confidence = record.getConfidence();
    }
}