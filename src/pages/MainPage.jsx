import { DUMMY_VIOLATIONS, DUMMY_STATS } from '../data/dummyData'
import { TYPE_COLOR, FEED_MAX_COUNT } from '../constants'
import { extractTime } from '../utils'
import StatCard from '../components/StatCard'
import styles from './MainPage.module.css'

// TODO: 백엔드 연결 시 아래 import 주석 해제 후 더미 데이터 교체
// import useApi from '../hooks/useApi'

const violations = DUMMY_VIOLATIONS
const stats = DUMMY_STATS
const connected = false // TODO: 백엔드 연결 시 useApi의 connected 사용

function MainPage() {
  // TODO: 백엔드 연결 시 아래 코드 사용
  // const { data: violations, loading, error, connected } = useApi('/api/violations?limit=10')
  // const { data: stats } = useApi('/api/stats')
  // if (loading) return <div className={styles.status}>불러오는 중...</div>
  // if (error) return <div className={styles.statusError}>연결 오류: {error}</div>

  return (
    <div className={styles.page}>
      {/* 영상 + 알림 영역 */}
      <div className={styles.topSection}>
        {/* 영상 스트림 */}
        <div className={styles.streamBox}>
          <div className={styles.streamHeader}>
            <span
              className={styles.liveDot}
              style={{ backgroundColor: connected ? '#22c55e' : '#64748b' }}
            />
            {connected ? 'LIVE' : 'OFFLINE'} &nbsp; CAM-01
          </div>
          {/* TODO: 실제 스트림 연결 시 아래 img 태그 사용 */}
          {/* <img src="http://AI서버주소/stream" className={styles.stream} alt="stream" /> */}
          <div className={styles.streamPlaceholder}>
            영상 스트림 영역<br />
            <small>백엔드 연결 후 활성화</small>
          </div>
        </div>

        {/* 실시간 알림 피드 */}
        <div className={styles.feedBox}>
          <div className={styles.feedTitle}>실시간 위반 알림</div>
          <ul className={styles.feedList}>
            {violations.slice(0, FEED_MAX_COUNT).map((v) => (
              <li key={v.id} className={styles.feedItem}>
                <span
                  className={styles.feedBadge}
                  style={{ backgroundColor: TYPE_COLOR[v.type] || '#64748b' }}
                >
                  {v.type}
                </span>
                <span className={styles.feedCamera}>{v.camera}</span>
                <span className={styles.feedTime}>{extractTime(v.timestamp)}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* 통계 카드 */}
      <div className={styles.statsSection}>
        <StatCard label="오늘 총 위반" value={stats.total} color="#002855" />
        <StatCard label="헬멧 미착용" value={stats.helmet} color="#b91c1c" />
        <StatCard label="인도 주행" value={stats.sidewalk} color="#7c3aed" />
        <StatCard label="다인 탑승" value={stats.multiRider} color="#b45309" />
      </div>
    </div>
  )
}

export default MainPage
