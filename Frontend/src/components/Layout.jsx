import { useLocation, useNavigate } from 'react-router-dom'
import useApi from '../hooks/useApi'
import styles from './Layout.module.css'

const NAV_GROUPS = [
  {
    label: 'OPERATIONS',
    items: [
      {
        path: '/',
        label: '모니터링',
        badge: 'LIVE',
        icon: (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="3" width="20" height="14" rx="2" />
            <line x1="8" y1="21" x2="16" y2="21" />
            <line x1="12" y1="17" x2="12" y2="21" />
          </svg>
        ),
      },
      {
        path: '/violations',
        label: '단속 기록',
        badge: 'violations',
        icon: (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
        ),
      },
      {
        path: '/stats',
        label: '통계',
        badge: null,
        icon: (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="20" x2="18" y2="10" />
            <line x1="12" y1="20" x2="12" y2="4" />
            <line x1="6" y1="20" x2="6" y2="14" />
          </svg>
        ),
      },
    ],
  },
  {
    label: 'CONFIG',
    items: [
      {
        path: '/zones',
        label: '구역 설정',
        badge: 'zones',
        icon: (
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
            <circle cx="12" cy="10" r="3"/>
          </svg>
        ),
      },
    ],
  },
]

function Layout({ children }) {
  const location = useLocation()
  const navigate = useNavigate()

  // 사이드바 카운트 뱃지 및 AI 서버 상태 (30초 간격으로 가볍게 폴링)
  const { data: violationsData } = useApi('/api/violations?limit=9999', 30000)
  const { data: zonesData, connected: aiConnected } = useApi('/ai/zones', 30000)

  const violationCount = violationsData?.length != null
    ? violationsData.length.toLocaleString()
    : '—'
  const zoneCount = zonesData?.zones?.length != null
    ? String(zonesData.zones.length)
    : '—'

  function getBadge(item) {
    if (item.badge === 'LIVE') return 'LIVE'
    if (item.badge === 'violations') return violationCount
    if (item.badge === 'zones') return zoneCount
    return null
  }

  return (
    <div className={styles.container}>
      <aside className={styles.sidebar}>
        {/* 브랜드 */}
        <div className={styles.brand}>
          <svg
            className={styles.brandIcon}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.25"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="5" cy="19" r="2" />
            <circle cx="19" cy="19" r="2" />
            <line x1="5" y1="17" x2="19" y2="17" />
            <line x1="7" y1="5" x2="13" y2="5" />
            <path d="M 10.6875 6.875 L 8.1875 11 L 11.8125 11 L 9.3125 15.125" stroke="#f59e0b" strokeWidth="1.5" />
          </svg>
          <div className={styles.brandName}>
            <span className={styles.brandTitle}>전동킥보드</span>
            <span className={styles.brandSub}>단속 시스템</span>
          </div>
        </div>

        {/* 네비게이션 */}
        <nav className={styles.nav}>
          {NAV_GROUPS.map((group, gi) => (
            <div key={group.label} className={styles.navGroup}>
              <div className={`${styles.navSection} ${gi === 0 ? styles.navSectionFirst : ''}`}>
                {group.label}
              </div>
              {group.items.map(({ path, label, icon }) => {
                const isActive = location.pathname === path
                const badge = getBadge(NAV_GROUPS[gi].items.find(i => i.path === path))
                return (
                  <button
                    key={path}
                    className={`${styles.navItem} ${isActive ? styles.active : ''}`}
                    onClick={() => navigate(path)}
                  >
                    {icon}
                    {label}
                    {badge && (
                      <span className={`${styles.navBadge} ${badge === 'LIVE' ? styles.navBadgeLive : ''}`}>
                        {badge}
                      </span>
                    )}
                  </button>
                )
              })}
            </div>
          ))}
        </nav>

        {/* 하단 AI 서버 상태 */}
        <div className={styles.sideFoot}>
          <span className={`${styles.sideFootDot} ${aiConnected ? styles.aiOn : styles.aiOff}`} />
          <span>{aiConnected ? 'AI 서버 연결됨' : 'AI 서버 미연결'}</span>
        </div>
      </aside>

      <main className={styles.main}>{children}</main>
    </div>
  )
}

export default Layout
