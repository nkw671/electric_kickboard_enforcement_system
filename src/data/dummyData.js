// 백엔드 연결 전 테스트용 더미 데이터

export const DUMMY_VIOLATIONS = [
  {
    id: 1,
    type: '헬멧 미착용',
    camera: 'CAM-01',
    timestamp: '2025-03-28 14:32:01',
    confidence: 94,
    image: null,
  },
  {
    id: 2,
    type: '인도 주행',
    camera: 'CAM-02',
    timestamp: '2025-03-28 14:28:55',
    confidence: 88,
    image: null,
  },
  {
    id: 3,
    type: '다인 탑승',
    camera: 'CAM-01',
    timestamp: '2025-03-28 14:15:43',
    confidence: 91,
    image: null,
  },
  {
    id: 4,
    type: '헬멧 미착용',
    camera: 'CAM-03',
    timestamp: '2025-03-28 13:58:20',
    confidence: 96,
    image: null,
  },
  {
    id: 5,
    type: '인도 주행',
    camera: 'CAM-01',
    timestamp: '2025-03-28 13:40:11',
    confidence: 85,
    image: null,
  },
  {
    id: 6,
    type: '헬멧 미착용',
    camera: 'CAM-02',
    timestamp: '2025-03-28 13:22:30',
    confidence: 92,
    image: null,
  },
]

export const DUMMY_STATS = {
  total: 6,
  helmet: 3,
  sidewalk: 2,
  multiRider: 1,
}
