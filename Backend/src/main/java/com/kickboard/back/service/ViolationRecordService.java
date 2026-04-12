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

import java.util.List;
import java.util.stream.Collectors;
import java.util.Map;
import java.util.HashMap;

// 단속 데이터의 DB 저장 및 프론트엔드 조회를 위한 비즈니스 로직 처리 클래스
@Service
@RequiredArgsConstructor
public class ViolationRecordService {

    private final ViolationRecordRepository repository;

    public void saveViolation(ViolationCreateRequest request) {
        ViolationRecord record = new ViolationRecord();
        record.setViolationType(request.getViolationType());
        record.setImageUrl(request.getImageUrl());
        record.setCamera(request.getCamera());
        record.setConfidence(request.getConfidence());

        repository.save(record);
    }

    public List<ViolationResponse> getRecentViolations(int limit) {
        Pageable pageable = PageRequest.of(0, limit, Sort.by(Sort.Direction.DESC, "createdAt"));
        List<ViolationRecord> records = repository.findAll(pageable).getContent();

        return records.stream()
                .map(ViolationResponse::new)
                .collect(Collectors.toList());
    }

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