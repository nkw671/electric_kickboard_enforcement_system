package com.kickboard.back.repository;

import com.kickboard.back.entity.ViolationRecord;
import org.springframework.data.jpa.repository.JpaRepository;

public interface ViolationRecordRepository extends JpaRepository<ViolationRecord, Long> {

}