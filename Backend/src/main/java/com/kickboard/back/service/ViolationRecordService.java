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

    // AI 단속 데이터 DB 저장 및 연결된 클라이언트에 실시간 브로드캐스트
    public void saveViolation(ViolationCreateRequest request) {
        ViolationRecord record = new ViolationRecord();
        record.setViolationType(request.getViolationType());
        record.setImageUrl(request.getImageUrl());
        record.setCamera(request.getCamera());
        record.setConfidence(request.getConfidence());

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

    // 프론트엔드 요청 개수에 따른 최신 단속 기록 조회
    public List<ViolationResponse> getRecentViolations(int limit) {
        Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "createdAt"));
        List<ViolationRecord> records = repository.findAll(pageable).getContent();
        return records.stream().map(ViolationResponse::new).collect(Collectors.toList());
    }

    // 대시보드 표시용 위반 유형별 누적 통계 산출
    public Map<String, Integer> getStats() {
        Map<String, Integer> stats = new HashMap<>();
        long total = repository.count();
        long helmet = repository.countByViolationType("헬멧 미착용");
        long sidewalk = repository.countByViolationType("인도 주행");
        long multiRider = repository.countByViolationType("다인 탑승");

        stats.put("total", (int) total);
        stats.put("helmet", (int) helmet);
        stats.put("sidewalk", (int) sidewalk);
        stats.put("multiRider", (int) multiRider);
        return stats;
    }
}