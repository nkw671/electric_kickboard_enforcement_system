/**
 * "YYYY-MM-DD HH:MM:SS" 형식의 타임스탬프에서 시간 부분만 반환
 * 형식이 다를 경우 원본 문자열을 그대로 반환
 */
export function extractTime(timestamp) {
  if (!timestamp) return ''
  const parts = timestamp.split(' ')
  return parts.length === 2 ? parts[1] : timestamp
}
