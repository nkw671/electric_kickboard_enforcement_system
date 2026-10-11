package com.kickboard.back.dto;

import lombok.Builder;
import lombok.Getter;
import java.util.List;

// 프론트엔드에 복합적인 통계 데이터를 전달할 때 사용하는 응답용 DTO
@Getter
@Builder
public class DashboardStatsResponse {
    private OverallStats overall;
    private List<HourlyStat> hourly;
    private List<LocationStat> topLocations;

    @Getter
    @Builder
    public static class OverallStats {
        private long total;
        private long helmet;
        private long sidewalk;
        private long multiRider;
    }

    @Getter
    @Builder
    public static class HourlyStat {
        private int hour;
        private long count;
    }

    @Getter
    @Builder
    public static class LocationStat {
        private String location;
        private long count;
    }
}