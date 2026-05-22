package com.kickboard.back.repository;

import com.kickboard.back.entity.ViolationRecord;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import java.time.LocalDateTime;
import java.util.List;

// MySQL DB의 단속 데이터에 접근하여 CRUD 및 통계 쿼리를 수행하는 인터페이스
public interface ViolationRecordRepository extends JpaRepository<ViolationRecord, Long> {

    // ==========================================
    // 대시보드 통계용 쿼리 (개수 산출)
    // ==========================================

    // 1. 특정 위반 유형에 일치하는 데이터의 총 개수를 반환
    long countByViolationType(String violationType);

    // 2. 특정 기간 동안 발생한 전체 단속 데이터의 총 개수를 반환
    long countByCreatedAtBetween(LocalDateTime start, LocalDateTime end);

    // 3. 특정 기간 동안 발생한 특정 위반 유형의 단속 데이터 개수를 반환
    long countByViolationTypeAndCreatedAtBetween(String violationType, LocalDateTime start, LocalDateTime end);

    // 4. 특정 구역에서 발생한 전체 단속 데이터의 총 개수를 반환
    long countByCamera(String camera);

    // 5. 특정 기간 동안 특정 구역에서 발생한 전체 단속 데이터 개수를 반환
    long countByCameraAndCreatedAtBetween(String camera, LocalDateTime start, LocalDateTime end);

    // 6. 특정 구역에서 발생한 특정 위반 유형의 데이터 개수를 반환
    long countByViolationTypeAndCamera(String violationType, String camera);

    // 7. 특정 기간 동안 특정 구역에서 발생한 특정 위반 유형의 데이터 개수를 반환
    long countByViolationTypeAndCameraAndCreatedAtBetween(String violationType, String camera, LocalDateTime start, LocalDateTime end);


    // ==========================================
    // 위반 기록 조회용 쿼리 (데이터 목록 필터링)
    // ==========================================

    // 1. 특정 위반 유형으로 필터링하여 데이터 목록 조회
    List<ViolationRecord> findByViolationType(String violationType, Pageable pageable);

    // 2. 특정 구역으로 필터링하여 데이터 목록 조회
    List<ViolationRecord> findByCamera(String camera, Pageable pageable);

    // 3. 특정 위반 유형과 특정 구역 조건을 모두 만족하는 데이터 목록 조회
    List<ViolationRecord> findByViolationTypeAndCamera(String violationType, String camera, Pageable pageable);
}