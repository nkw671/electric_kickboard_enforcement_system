package com.kickboard.back.dto;

import lombok.Builder;
import lombok.Getter;

import java.time.LocalDateTime;
import java.util.Map;

// 프론트엔드가 일관되게 에러를 처리할 수 있도록 제공하는 공통 에러 규격
@Getter
@Builder
public class ErrorResponse {

    @Builder.Default
    private LocalDateTime timestamp = LocalDateTime.now(); // 에러 발생 시간

    private int status; // HTTP 상태 코드

    private String errorCode; // 프론트엔드 분기 처리용 커스텀 에러 코드

    private String message; // 사용자에게 보여줄 수 있는 일반적인 에러 메시지

    // (선택) 입력값 검증 실패 등 상세한 필드별 에러 원인이 있을 경우 담는 맵
    private Map<String, String> errors;
}