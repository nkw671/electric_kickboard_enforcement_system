package com.kickboard.back.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

// AI 서버에서 전송하는 위반 데이터를 담아 백엔드로 전달하기 위한 DTO
@Getter
@Setter
@NoArgsConstructor
public class ViolationCreateRequest {

    @JsonProperty("type")
    private String violationType; // 위반 종류

    @JsonProperty("image_url")
    private String imageUrl; // 위반 이미지 주소

    private String camera; // 카메라 번호
    private Integer confidence; // AI 정확도

}