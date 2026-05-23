package com.kickboard.back.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;

// AI 서버에서 전송하는 위반 데이터를 담아 백엔드로 전달하기 위한 DTO
@Getter
@Setter
@NoArgsConstructor
public class ViolationCreateRequest {

    @NotBlank(message = "위반 유형(type)은 필수 입력값입니다.")
    @JsonProperty("type")
    private String violationType;

    @NotBlank(message = "사진 URL(image_url)은 필수 입력값입니다.")
    @JsonProperty("image_url")
    private String imageUrl;

    @NotBlank(message = "카메라 번호(camera)는 필수 입력값입니다.")
    private String camera;

    @NotNull(message = "신뢰도(confidence)는 필수 입력값입니다.")
    @Min(value = 0, message = "신뢰도는 0 이상이어야 합니다.")
    @Max(value = 100, message = "신뢰도는 100 이하여야 합니다.")
    private Integer confidence;

    private String location;

}