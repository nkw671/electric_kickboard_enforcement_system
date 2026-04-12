package com.kickboard.back.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.time.LocalDateTime;

// 데이터베이스의 'violation_record' 테이블과 1:1 매핑되어 단속 정보를 저장하는 엔티티 클래스
@Entity
@Table(name = "violation_record")
@Getter
@Setter
@NoArgsConstructor
public class ViolationRecord {

    @Id // Primary Key
    @GeneratedValue(strategy = GenerationType.IDENTITY) // 데이터 번호
    private Long id;

    @Column(nullable = false)
    private String violationType; // 위반 종류

    @Column(length = 500)
    private String imageUrl; // 위반 이미지 주소

    @Column
    private String camera; // 카메라 번호

    @Column
    private Integer confidence; // AI 정확도

    @Column(nullable = false)
    private LocalDateTime createdAt; // 위반 발생 시간

    @PrePersist
    protected void onCreate() {
        this.createdAt = LocalDateTime.now();
    }
}