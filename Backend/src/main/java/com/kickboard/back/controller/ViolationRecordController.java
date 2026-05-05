package com.kickboard.back.controller;

import com.kickboard.back.dto.ViolationCreateRequest;
import com.kickboard.back.dto.ViolationResponse;
import com.kickboard.back.service.ViolationRecordService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.List;
import java.util.Map;

// AI 데이터 수신(POST) 및 프론트엔드 데이터 제공(GET)을 담당하는 API 엔드포인트 클래스
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
public class ViolationRecordController {

    private final ViolationRecordService service;

    // AI 서버로부터 단속 데이터 수신 (POST /api/violations)
    @PostMapping("/violations")
    public String receiveViolation(@RequestBody ViolationCreateRequest request) {
        service.saveViolation(request);
        return "단속 데이터 저장 성공";
    }

    // 프론트엔드 단속 기록 데이터 제공 (GET /api/violations)
    @GetMapping("/violations")
    public List<ViolationResponse> getViolations(
            @RequestParam(defaultValue = "10") int limit
    ) {
        return service.getRecentViolations(limit);
    }

    // 프론트엔드 대시보드용 위반 통계 데이터 제공 (GET /api/stats)
    @GetMapping("/stats")
    public Map<String, Integer> getStats() {
        return service.getStats();
    }

    // 프론트엔드 실시간 알림(SSE) 수신용 스트림 연결 (GET /api/stream)
    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter subscribe() {
        return service.subscribe();
    }

}