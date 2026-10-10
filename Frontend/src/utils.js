/**
 * "YYYY-MM-DD HH:MM:SS" 형식의 타임스탬프에서 시간 부분만 반환
 * 형식이 다를 경우 원본 문자열을 그대로 반환
 */
export function extractTime(timestamp) {
  if (!timestamp) return ''
  const parts = timestamp.split(' ')
  return parts.length === 2 ? parts[1] : timestamp
}

/**
 * /api/violations 응답에서 위반 목록 배열만 꺼낸다.
 * - 배열이면 그대로 (기존 응답 형태)
 * - Spring Page 객체({ content, totalElements, ... })면 content를 꺼냄 (백엔드 페이지네이션 적용 후)
 * 백엔드 머지 후 배열 분기는 제거해도 됨.
 */
export function toViolationList(data) {
  if (Array.isArray(data)) return data
  return data?.content ?? []
}

/**
 * 로컬 타임존 기준 오늘 날짜를 "YYYY-MM-DD" 형식으로 반환
 * (Date#toISOString은 UTC 변환이라 자정 근처에 날짜가 밀릴 수 있어 사용하지 않음)
 */
export function getTodayDateString() {
  const now = new Date()
  const y = now.getFullYear()
  const m = String(now.getMonth() + 1).padStart(2, '0')
  const d = String(now.getDate()).padStart(2, '0')
  return `${y}-${m}-${d}`
}
