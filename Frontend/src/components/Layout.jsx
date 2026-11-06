import { useLocation, useNavigate } from 'react-router-dom';
import { DEMO_MODE } from '../demo/demoMode';
import styles from './Layout.module.css';

// 메뉴 아이콘: Lucide 아이콘(ISC 라이선스) 모양을 그대로 옮겨 씀. 선 두께는 글자와 어울리게 2.25
function NavIcon({ children }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.25" strokeLinecap="round" strokeLinejoin="round">
      {children}
    </svg>
  );
}

const NAV_ITEMS = [
  {
    path: '/',
    label: '모니터링',
    // Lucide: video
    icon: (
      <NavIcon>
        <path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.87a.5.5 0 0 0-.752-.432L16 10.5" />
        <rect x="2" y="6" width="14" height="12" rx="2" />
      </NavIcon>
    ),
  },
  {
    path: '/violations',
    label: '단속 기록',
    // Lucide: file-text
    icon: (
      <NavIcon>
        <path d="M15 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7Z" />
        <path d="M14 2v4a2 2 0 0 0 2 2h4" />
        <path d="M10 9H8" />
        <path d="M16 13H8" />
        <path d="M16 17H8" />
      </NavIcon>
    ),
  },
  {
    path: '/stats',
    label: '통계',
    // Lucide: chart-column (구 bar-chart-3)
    icon: (
      <NavIcon>
        <path d="M3 3v16a2 2 0 0 0 2 2h16" />
        <path d="M18 17V9" />
        <path d="M13 17V5" />
        <path d="M8 17v-3" />
      </NavIcon>
    ),
  },
  {
    path: '/zones',
    label: '구역 설정',
    // Lucide: map-pin
    icon: (
      <NavIcon>
        <path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0" />
        <circle cx="12" cy="10" r="3" />
      </NavIcon>
    ),
  },
];

function Layout({ children }) {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <div className={styles.container}>
      <aside className={styles.sidebar}>
        <div className={styles.brand}>
          {/* 번개 아이콘: "Thunder" by khaerul ilyas, Noun Project (CC BY 3.0) */}
          <span className={styles.brandMark}>
            <svg className={styles.brandIcon} viewBox="18 10 66 90" fill="currentColor" aria-hidden="true">
              <path d="m21.969 63.406 17.371-2.2969c0.10156-0.011719 0.20312-0.019531 0.30859-0.019531 0.64062 0 1.2578 0.26172 1.7031 0.73438 0.51563 0.54688 0.73828 1.3086 0.59766 2.0508l-12.785 33.777 42.066-43.98-17.52-0.003906c-0.89453 0-1.707-0.50781-2.1016-1.3086s-0.30078-1.7578 0.24609-2.4648l24.441-31.773-47.191 13.539z" />
            </svg>
          </span>
          <span className={styles.brandName}>
            <span className={styles.brandTitle}>KickDan</span>
          </span>
        </div>
        <nav className={styles.nav}>
          {NAV_ITEMS.map(({ path, label, icon }) => {
            const isActive = location.pathname === path;
            return (
              <button
                key={path}
                className={`${styles.navItem} ${isActive ? styles.active : ''}`}
                onClick={() => navigate(path)}
              >
                {icon}
                {label}
              </button>
            );
          })}
        </nav>
        {DEMO_MODE && (
          <div className={styles.demoBadge} title="서버 대신 예시 데이터를 보여주는 중 (?demo=0 으로 끄기)">
            데모 모드
          </div>
        )}
      </aside>
      <main className={styles.main}>{children}</main>
    </div>
  );
}

export default Layout;
