package com.kickboard.back.controller;

import com.kickboard.back.dto.ViolationCreateRequest;
import com.kickboard.back.dto.ViolationResponse;
import com.kickboard.back.service.ViolationRecordService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;
import java.util.HashMap;

// AI 데이터 수신(POST) 및 프론트엔드 데이터 제공(GET)을 담당하는 API 엔드포인트 클래스
@RestController
@RequestMapping("/api")
@RequiredArgsConstructor
@CrossOrigin(origins = "http://localhost:5173")
public class ViolationRecordController {

    private final ViolationRecordService service;

    // AI 데이터 수신 (POST /api/violations)
    @PostMapping("/violations")
    public String receiveViolation(@RequestBody ViolationCreateRequest request) {
        service.saveViolation(request);
        return "단속 데이터 저장 성공";
    }

    // 프론트엔드 데이터 제공 (GET /api/violations)
    @GetMapping("/violations")
    public List<ViolationResponse> getViolations(
            @RequestParam(defaultValue = "10") int limit
    ) {
        return service.getRecentViolations(limit);
    }

    // 임시 통계 데이터 제공 (GET /api/stats)
    @GetMapping("/stats")
    public Map<String, Integer> getStats() {
        Map<String, Integer> stats = new HashMap<>();
        stats.put("total", 12);
        stats.put("helmet", 7);
        stats.put("sidewalk", 3);
        stats.put("multiRider", 2);
        return stats;
    }
}