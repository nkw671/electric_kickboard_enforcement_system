/**
 * 데모 모드 여부. 발표 중 AI/백엔드 서버가 꺼져 있어도 화면을 보여주기 위한 것.
 * - VITE_DEMO_MODE=true 로 빌드하면 항상 켜짐
 * - 주소에 ?demo=1 을 붙이면 켜지고(탭을 닫을 때까지 유지), ?demo=0 이면 꺼짐
 */
const STORAGE_KEY = 'kickdan-demo'

function readDemoMode() {
  if (import.meta.env.VITE_DEMO_MODE === 'true') return true
  const param = new URLSearchParams(window.location.search).get('demo')
  try {
    if (param === '1') sessionStorage.setItem(STORAGE_KEY, '1')
    if (param === '0') sessionStorage.removeItem(STORAGE_KEY)
    return sessionStorage.getItem(STORAGE_KEY) === '1'
  } catch {
    // 시크릿 모드 등으로 sessionStorage를 못 쓰면 쿼리 파라미터만 본다.
    return param === '1'
  }
}

export const DEMO_MODE = readDemoMode()
