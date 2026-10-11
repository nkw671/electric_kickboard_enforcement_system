package com.kickboard.back.service;

import com.kickboard.back.dto.DashboardStatsResponse;
import com.kickboard.back.dto.ViolationCreateRequest;
import com.kickboard.back.dto.ViolationResponse;
import com.kickboard.back.entity.ViolationRecord;
import com.kickboard.back.repository.ViolationRecordRepository;
import com.kickboard.back.repository.ViolationRecordRepository.HourlyStatProjection;
import com.kickboard.back.repository.ViolationRecordRepository.LocationStatProjection;
import com.kickboard.back.repository.ViolationRecordRepository.OverallStatProjection;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.List;
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
    // 단속 기록 목록 조회 (필터링 및 페이징 적용)
    // ==========================================

    // 프론트엔드 요청 조건(유형, 구역, 개수)에 따른 단속 기록 조회
    public Page<ViolationResponse> getRecentViolations(String type, String camera, int page, int size) {
        Pageable pageable = PageRequest.of(page, size, Sort.by(Sort.Direction.DESC, "createdAt"));
        Page<ViolationRecord> recordPage;

        // 필터링 조건 존재 여부 확인
        boolean hasType = (type != null && !type.isEmpty() && !type.equals("전체"));
        boolean hasCamera = (camera != null && !camera.isEmpty() && !camera.equals("전체"));
        // 1. 유형과 구역 조건이 모두 있을 때
        if (hasType && hasCamera) {
            recordPage = repository.findByViolationTypeAndCamera(type, camera, pageable);
        }
        // 2. 유형 조건만 있을 때
        else if (hasType) {
            recordPage = repository.findByViolationType(type, pageable);
        }
        // 3. 구역 조건만 있을 때

        else if (hasCamera) {
            recordPage = repository.findByCamera(camera, pageable);
        }
        // 4. 조건이 없거나 "전체"일 때
        else {
            recordPage = repository.findAll(pageable);
        }

        // Page 객체 내부의 Entity들을 DTO로 변환하여 반환
        return recordPage.map(ViolationResponse::new);
    }

    // ==========================================
    // 대시보드 통계 산출 (단일 쿼리 및 다차원 집계로 최적화)
    // ==========================================

    // 기간별 및 구역별 누적 통계 산출 (전체, 시간대별, 위치 TOP 5)
    public DashboardStatsResponse getStats(LocalDate startDate, LocalDate endDate, String camera) {
        // 1. 파라미터 전처리
        LocalDateTime start = (startDate != null) ? startDate.atStartOfDay() : null;
        LocalDateTime end = (endDate != null) ? endDate.atTime(LocalTime.MAX) : null;
        String cam = (camera != null && !camera.isEmpty() && !camera.equals("전체")) ? camera : null;

        // 2. 각각의 통계 데이터를 DB에서 조회
        OverallStatProjection overallProj = repository.getOverallStats(start, end, cam);
        List<HourlyStatProjection> hourlyProj = repository.getHourlyStats(start, end, cam);
        List<LocationStatProjection> locationProj = repository.getTopLocations(start, end, cam, PageRequest.of(0, 5)); // TOP 5 추출

        // 3. DTO 조립 및 반환 (Null-safe 처리)
        return DashboardStatsResponse.builder()
                .overall(DashboardStatsResponse.OverallStats.builder()
                        .total(overallProj != null && overallProj.getTotal() != null ? overallProj.getTotal() : 0)
                        .helmet(overallProj != null && overallProj.getHelmet() != null ? overallProj.getHelmet() : 0)
                        .sidewalk(overallProj != null && overallProj.getSidewalk() != null ? overallProj.getSidewalk() : 0)
                        .multiRider(overallProj != null && overallProj.getMultiRider() != null ? overallProj.getMultiRider() : 0)
                        .build())
                .hourly(hourlyProj.stream()
                        .map(h -> DashboardStatsResponse.HourlyStat.builder()
                                .hour(h.getHour())
                                .count(h.getCount())
                                .build())
                        .collect(Collectors.toList()))
                .topLocations(locationProj.stream()
                        .map(l -> DashboardStatsResponse.LocationStat.builder()
                                .location(l.getLocation())
                                .count(l.getCount())
                                .build())
                        .collect(Collectors.toList()))
                .build();
    }
}