package com.kickboard.back.repository;

import com.kickboard.back.entity.ViolationRecord;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.time.LocalDateTime;
import java.util.List;

// MySQL DB의 단속 데이터에 접근하여 CRUD 및 통계 쿼리를 수행하는 인터페이스
public interface ViolationRecordRepository extends JpaRepository<ViolationRecord, Long> {

    // --- Projections (DB 조회 결과를 매핑할 인터페이스) ---
    interface OverallStatProjection {
        Long getTotal();
        Long getHelmet();
        Long getSidewalk();
        Long getMultiRider();
    }

    interface HourlyStatProjection {
        Integer getHour();
        Long getCount();
    }

    interface LocationStatProjection {
        String getLocation();
        Long getCount();
    }

    // --- 통계 최적화 쿼리 ---

    // 1. 전체 통계
    @Query("SELECT COUNT(v) as total, " +
            "COALESCE(SUM(CASE WHEN v.violationType = '헬멧 미착용' THEN 1 ELSE 0 END), 0) as helmet, " +
            "COALESCE(SUM(CASE WHEN v.violationType = '인도 주행' THEN 1 ELSE 0 END), 0) as sidewalk, " +
            "COALESCE(SUM(CASE WHEN v.violationType = '다인 탑승' THEN 1 ELSE 0 END), 0) as multiRider " +
            "FROM ViolationRecord v " +
            "WHERE (:start IS NULL OR v.createdAt >= :start) " +
            "AND (:end IS NULL OR v.createdAt <= :end) " +
            "AND (:camera IS NULL OR :camera = '전체' OR v.camera = :camera)")
    OverallStatProjection getOverallStats(@Param("start") LocalDateTime start, @Param("end") LocalDateTime end, @Param("camera") String camera);

    // 2. 시간대별 통계
    @Query("SELECT FUNCTION('HOUR', v.createdAt) as hour, COUNT(v) as count " +
            "FROM ViolationRecord v " +
            "WHERE (:start IS NULL OR v.createdAt >= :start) " +
            "AND (:end IS NULL OR v.createdAt <= :end) " +
            "AND (:camera IS NULL OR :camera = '전체' OR v.camera = :camera) " +
            "GROUP BY FUNCTION('HOUR', v.createdAt) " +
            "ORDER BY hour ASC")
    List<HourlyStatProjection> getHourlyStats(@Param("start") LocalDateTime start, @Param("end") LocalDateTime end, @Param("camera") String camera);

    // 3. 위치별 TOP N 통계
    @Query("SELECT v.location as location, COUNT(v) as count " +
            "FROM ViolationRecord v " +
            "WHERE (:start IS NULL OR v.createdAt >= :start) " +
            "AND (:end IS NULL OR v.createdAt <= :end) " +
            "AND (:camera IS NULL OR :camera = '전체' OR v.camera = :camera) " +
            "AND v.location IS NOT NULL " +
            "GROUP BY v.location " +
            "ORDER BY count DESC")
    List<LocationStatProjection> getTopLocations(@Param("start") LocalDateTime start, @Param("end") LocalDateTime end, @Param("camera") String camera, Pageable pageable);

    // --- 위반 기록 조회용 쿼리 (데이터 목록 필터링 및 페이징) ---

    // 1. 특정 위반 유형으로 필터링하여 데이터 목록 조회
    Page<ViolationRecord> findByViolationType(String violationType, Pageable pageable);
    // 2. 특정 구역으로 필터링하여 데이터 목록 조회
    Page<ViolationRecord> findByCamera(String camera, Pageable pageable);
    // 3. 특정 위반 유형과 특정 구역 조건을 모두 만족하는 데이터 목록 조회
    Page<ViolationRecord> findByViolationTypeAndCamera(String violationType, String camera, Pageable pageable);
}