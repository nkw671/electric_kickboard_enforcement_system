package com.kickboard.back.repository;

import com.kickboard.back.entity.ViolationRecord;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

// MySQL DB의 단속 데이터에 접근하여 CRUD 및 통계 쿼리를 수행하는 인터페이스
public interface ViolationRecordRepository extends JpaRepository<ViolationRecord, Long> {

    // 위반 유형에 일치하는 데이터의 총 개수를 반환
    long countByViolationType(String violationType);

    // 위반 유형으로 필터링해서 찾기
    List<ViolationRecord> findByViolationType(String violationType, Pageable pageable);
}