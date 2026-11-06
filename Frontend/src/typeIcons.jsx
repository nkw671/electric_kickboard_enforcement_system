// 위반 유형별 아이콘. Lucide 아이콘(ISC 라이선스) 모양을 그대로 옮겨 씀(헬멧만 직접 그림).
// 통계 요약 카드, 단속 기록 상세 창 배지에서 함께 사용한다.

// 컴포넌트가 아닌 함수로 둠 (상수만 내보내는 파일이라 Fast Refresh 규칙상)
function icon(children) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      {children}
    </svg>
  )
}

export const TYPE_ICON = {
  // 자전거·스포츠 헬멧 (Lucide에 없어서 같은 규격으로 직접 그림: 둥근 몸체 + 통풍 슬롯 2개 + 반원 턱끈)
  '헬멧 미착용': icon(<>
    <path d="M3 14a9 9 0 0 1 18 0H3Z" />
    <path d="M10 5.3 10.5 10" />
    <path d="M14 5.3 13.5 10" />
    <path d="M6 14a6 6 0 0 0 12 0" />
  </>),
  // Lucide: users
  '다인 탑승': icon(<>
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </>),
  // Lucide: footprints
  '인도 주행': icon(<>
    <path d="M4 16v-2.38C4 11.5 2.97 10.5 3 8c.03-2.72 1.49-6 4.5-6C9.37 2 10 3.8 10 5.5c0 3.11-2 5.66-2 8.68V16a2 2 0 1 1-4 0Z" />
    <path d="M20 20v-2.38c0-2.12 1.03-3.12 1-5.62-.03-2.72-1.49-6-4.5-6C14.63 6 14 7.8 14 9.5c0 3.11 2 5.66 2 8.68V20a2 2 0 1 0 4 0Z" />
    <path d="M16 17h4" />
    <path d="M4 13h4" />
  </>),
}

// 전체 단속 (Lucide: shield-check)
export const TOTAL_ICON = icon(<>
  <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z" />
  <path d="m9 12 2 2 4-4" />
</>)
