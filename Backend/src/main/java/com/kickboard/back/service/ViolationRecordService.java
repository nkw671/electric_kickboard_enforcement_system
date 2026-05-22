package com.kickboard.back.service;

import com.kickboard.back.dto.ViolationCreateRequest;
import com.kickboard.back.dto.ViolationResponse;
import com.kickboard.back.entity.ViolationRecord;
import com.kickboard.back.repository.ViolationRecordRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;
import java.util.Map;
import java.util.HashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

// 단속 데이터의 DB 저장, 조회, 통계 산출 및 실시간 알림 비즈니스 로직 처리 클래스
@Service
@RequiredArgsConstructor
public class ViolationRecordService {

    private final ViolationRecordRepository repository;
    private final List<SseEmitter> emitters = new CopyOnWriteArrayList<>();

    // ==========================================
    // 실시간 알림 (SSE) 관리
    // ==========================================

    // 프론트엔드 실시간 알림 파이프 연결 및 관리
    public SseEmitter subscribe() {
        SseEmitter emitter = new SseEmitter(60 * 1000L * 60); // 1시간 동안 파이프 유지
        emitters.add(emitter);

        emitter.onCompletion(() -> emitters.remove(emitter));
        emitter.onTimeout(() -> emitters.remove(emitter));

        try {
            emitter.send(SseEmitter.event().name("connect").data("연결 성공"));
        } catch (Exception e) {
            emitters.remove(emitter);
        }

        return emitter;
    }

    // ==========================================
    // 단속 데이터 저장 및 방송
    // ==========================================

    // AI 단속 데이터 DB 저장 및 연결된 클라이언트에 실시간 브로드캐스트
    public void saveViolation(ViolationCreateRequest request) {
        ViolationRecord record = new ViolationRecord();
        record.setViolationType(request.getViolationType());
        record.setImageUrl(request.getImageUrl());
        record.setCamera(request.getCamera());
        record.setConfidence(request.getConfidence());
        record.setLocation(request.getLocation());

        ViolationRecord savedRecord = repository.save(record);
        ViolationResponse response = new ViolationResponse(savedRecord);

        for (SseEmitter emitter : emitters) {
            try {
                emitter.send(SseEmitter.event()
                        .name("violation")
                        .data(response));
            } catch (Exception e) {
                emitters.remove(emitter);
            }
        }
    }

    // ==========================================
    // 단속 기록 목록 조회 (필터링 적용)
    // ==========================================

    // 프론트엔드 요청 조건(유형, 구역, 개수)에 따른 단속 기록 조회
    public List<ViolationResponse> getRecentViolations(String type, String camera, int limit) {
        Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "createdAt"));
        List<ViolationRecord> records;

        // 필터링 조건 존재 여부 확인
        boolean hasType = (type != null && !type.isEmpty() && !type.equals("전체"));
        boolean hasCamera = (camera != null && !camera.isEmpty() && !camera.equals("전체"));

        // 1. 유형과 구역 조건이 모두 있을 때
        if (hasType && hasCamera) {
            records = repository.findByViolationTypeAndCamera(type, camera, pageable);
        }
        // 2. 유형 조건만 있을 때
        else if (hasType) {
            records = repository.findByViolationType(type, pageable);
        }
        // 3. 구역 조건만 있을 때
        else if (hasCamera) {
            records = repository.findByCamera(camera, pageable);
        }
        // 4. 조건이 없거나 "전체"일 때
        else {
            records = repository.findAll(pageable).getContent();
        }

        return records.stream().map(ViolationResponse::new).collect(Collectors.toList());
    }

    // ==========================================
    // 대시보드 통계 산출 (필터링 적용)
    // ==========================================

    // 기간별 및 구역별 누적 통계 산출
    public Map<String, Integer> getStats(LocalDate startDate, LocalDate endDate, String camera) {
        long total, helmet, sidewalk, multiRider;

        // 필터링 조건 존재 여부 및 날짜 포맷 변환
        boolean hasDate = (startDate != null && endDate != null);
        boolean hasCamera = (camera != null && !camera.isEmpty() && !camera.equals("전체"));

        LocalDateTime start = hasDate ? startDate.atStartOfDay() : null;
        LocalDateTime end = hasDate ? endDate.atTime(LocalTime.MAX) : null;

        // 1. 기간과 구역 조건이 모두 있을 때
        if (hasDate && hasCamera) {
            total = repository.countByCameraAndCreatedAtBetween(camera, start, end);
            helmet = repository.countByViolationTypeAndCameraAndCreatedAtBetween("헬멧 미착용", camera, start, end);
            sidewalk = repository.countByViolationTypeAndCameraAndCreatedAtBetween("인도 주행", camera, start, end);
            multiRider = repository.countByViolationTypeAndCameraAndCreatedAtBetween("다인 탑승", camera, start, end);
        }
        // 2. 기간 조건만 있을 때
        else if (hasDate) {
            total = repository.countByCreatedAtBetween(start, end);
            helmet = repository.countByViolationTypeAndCreatedAtBetween("헬멧 미착용", start, end);
            sidewalk = repository.countByViolationTypeAndCreatedAtBetween("인도 주행", start, end);
            multiRider = repository.countByViolationTypeAndCreatedAtBetween("다인 탑승", start, end);
        }
        // 3. 구역 조건만 있을 때
        else if (hasCamera) {
            total = repository.countByCamera(camera);
            helmet = repository.countByViolationTypeAndCamera("헬멧 미착용", camera);
            sidewalk = repository.countByViolationTypeAndCamera("인도 주행", camera);
            multiRider = repository.countByViolationTypeAndCamera("다인 탑승", camera);
        }
        // 4. 조건이 없을 때 (전체 누적 통계)
        else {
            total = repository.count();
            helmet = repository.countByViolationType("헬멧 미착용");
            sidewalk = repository.countByViolationType("인도 주행");
            multiRider = repository.countByViolationType("다인 탑승");
        }

        Map<String, Integer> stats = new HashMap<>();
        stats.put("total", (int) total);
        stats.put("helmet", (int) helmet);
        stats.put("sidewalk", (int) sidewalk);
        stats.put("multiRider", (int) multiRider);
        return stats;
    }
}