package com.kickboard.back.controller;

import com.kickboard.back.dto.ViolationCreateRequest;
import com.kickboard.back.dto.ViolationResponse;
import com.kickboard.back.service.ViolationRecordService;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.time.LocalDate;
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
    // 유형(type)과 구역(camera) 조건을 조합하여 최신 기록을 조회합니다.
    @GetMapping("/violations")
    public List<ViolationResponse> getViolations(
            @RequestParam(required = false) String type,
            @RequestParam(required = false) String camera,
            @RequestParam(defaultValue = "10") int limit
    ) {
        return service.getRecentViolations(type, camera, limit);
    }

    // 프론트엔드 대시보드용 위반 통계 데이터 제공 (GET /api/stats)
    // 기간(startDate~endDate) 및 구역(camera) 조건에 따른 통계를 반환합니다.
    @GetMapping("/stats")
    public Map<String, Integer> getStats(
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate startDate,
            @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE) LocalDate endDate,
            @RequestParam(required = false) String camera
    ) {
        return service.getStats(startDate, endDate, camera);
    }

    // 프론트엔드 실시간 알림(SSE) 수신용 스트림 연결 (GET /api/stream)
    @GetMapping(value = "/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter subscribe() {
        return service.subscribe();
    }

}