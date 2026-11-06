import { useMemo } from 'react'
import useApi from '../hooks/useApi'
import { apiUrl } from '../config'
import { toViolationList } from '../utils'
import styles from './StatsPage.module.css'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts'

const TYPE_ICON = {
  '헬멧 미착용': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a9 9 0 0 0-9 9h18a9 9 0 0 0-9-9z"/>
      <path d="M3 11h18v2a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-2z"/>
      <path d="M8 15v1a4 4 0 0 0 8 0v-1"/>
    </svg>
  ),
  '다인 탑승': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/>
      <circle cx="9" cy="7" r="4"/>
      <path d="M22 21v-2a4 4 0 0 0-3-3.87"/>
      <path d="M16 3.13a4 4 0 0 1 0 7.75"/>
    </svg>
  ),
  '인도 주행': (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="4" r="2"/>
      <path d="M9 22v-7l-2-5h10l-2 5v7"/>
      <path d="M9 12h6"/>
    </svg>
  ),
}

const TYPE_CONFIG = {
  '헬멧 미착용': { color: '#f59e0b', borderColor: '#fde68a' },
  '다인 탑승':   { color: '#3b82f6', borderColor: '#bfdbfe' },
  '인도 주행':   { color: '#ef4444', borderColor: '#fecaca' },
}


function StatsPage() {
  // 요약 카드/파이 차트는 서버가 집계해주는 /api/stats를 그대로 쓴다.
  // 시간대별 차트만 백엔드에 시간대별 집계 API가 아직 없어서 전체 기록을
  // 받아 클라이언트에서 집계한다.
  // /api/stats: 백엔드 머지 후에는 { overall: {...} }, 그 전에는 평탄한 객체 — 둘 다 지원
  const { data: statsData, loading: statsLoading, error: statsError } = useApi(apiUrl('/api/stats'))
  const overall = statsData?.overall ?? statsData
  // limit(기존)과 size(페이지네이션 적용 후)를 같이 보냄 — 백엔드 머지 후 limit 제거
  const { data: apiData, loading: violationsLoading, error: violationsError } = useApi(apiUrl('/api/violations?limit=9999&size=9999'))
  const violations = useMemo(() => toViolationList(apiData), [apiData])

  // 두 API 중 하나만 실패하면 그 데이터를 쓰는 카드에만 에러를 표시한다.
  // 이전 데이터가 있으면 폴링 한 번 실패해도 그대로 보여준다.
  const statsFailed = statsError && !statsData
  const violationsFailed = violationsError && !apiData

  const total = overall?.total ?? 0

  // 폴링 응답은 값이 같아도 매번 새 객체라서, overall을 의존성으로 두면 3초마다
  // 차트 애니메이션이 처음부터 다시 재생된다. 값이 바뀔 때만 새 배열을 만든다.
  const helmet = overall?.helmet ?? 0
  const multiRider = overall?.multiRider ?? 0
  const sidewalk = overall?.sidewalk ?? 0
  const typeData = useMemo(() => [
    { name: '헬멧 미착용', value: helmet },
    { name: '다인 탑승', value: multiRider },
    { name: '인도 주행', value: sidewalk },
  ], [helmet, multiRider, sidewalk])

  // 0~23시 버킷을 항상 다 만들어 둔다. 실제로 기록이 있는 시간대만 버킷을 만들면
  // (특히 발표 데모처럼 데이터가 한두 시간대에 몰릴 때) 막대가 하나뿐이라 차트
  // 폭 전체를 꽉 채워버리는 문제가 있었음.
  const rawTimeData = useMemo(() => {
    const buckets = {}
    for (let h = 0; h < 24; h++) {
      const hour = String(h).padStart(2, '0') + '시'
      buckets[hour] = { time: hour, '헬멧 미착용': 0, '다인 탑승': 0, '인도 주행': 0 }
    }
    violations.forEach(v => {
      const hour = v.timestamp.slice(11, 13) + '시'
      if (v.type in (buckets[hour] || {})) buckets[hour][v.type]++
    })
    return Object.values(buckets)
  }, [violations])
  // 위와 같은 이유로, 집계 결과가 실제로 바뀔 때만 차트 데이터를 교체한다.
  const timeDataKey = JSON.stringify(rawTimeData)
  const timeData = useMemo(() => JSON.parse(timeDataKey), [timeDataKey])

  if (statsLoading && violationsLoading) return <div className={styles.status}>불러오는 중...</div>

  return (
    <div className={styles.page}>
      {/* 헤더 */}
      <div className={styles.header}>
        <h2 className={styles.title}>단속 통계</h2>
        <p className={styles.desc}>
          누적 단속 기록을 위반 유형별·시간대별로 집계합니다.
          <br />
          위반이 많은 시간대를 확인해 집중 단속 시간을 정하는 데 활용할 수 있습니다.
        </p>
      </div>

      <div className={styles.content}>
        {statsFailed ? (
          <div className={styles.cardError}>통계 서버에 연결할 수 없습니다.</div>
        ) : (
          <>
            {/* 요약 카드 4개 */}
            <div className={styles.summaryRow}>
              <div className={styles.summaryCard}>
                <div className={styles.summaryCardLabel}>
                  <svg className={styles.summaryIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    <line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  전체 단속
                </div>
                <p className={styles.summaryVal}>{total}</p>
                <p className={styles.summaryUnit}>건</p>
              </div>
              {typeData.map(({ name, value }) => {
                const cfg = TYPE_CONFIG[name]
                const pct = total ? Math.round((value / total) * 100) : 0
                return (
                  <div key={name} className={styles.summaryCard} style={{ borderColor: cfg.borderColor }}>
                    <div className={styles.summaryCardLabel} style={{ color: cfg.color }}>
                      {TYPE_ICON[name]}
                      {name}
                    </div>
                    <p className={styles.summaryVal}>{value}</p>
                    <p className={styles.summaryUnit}>건 ({pct}%)</p>
                  </div>
                )
              })}
            </div>

            {/* 위반 유형 분포 */}
            <div className={styles.chartCard}>
              <div className={styles.chartCardHeader}>
                <svg className={styles.chartHeaderIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                위반 유형 분포
              </div>
              <ResponsiveContainer width="100%" height={110} style={{ marginTop: '-8px' }}>
                <PieChart>
                  {/* 애니메이션을 끔: recharts Pie는 다시 렌더링될 때마다(3초 폴링 등) 애니메이션을
                      처음부터 다시 시작해서, 타이밍에 따라 몇 초 동안 빈 차트로 보이는 문제가 있었음 */}
                  <Pie data={typeData} cx="50%" cy="50%" innerRadius={34} outerRadius={55} paddingAngle={3} dataKey="value" isAnimationActive={false}>
                    {typeData.map(({ name }) => (
                      <Cell key={name} fill={TYPE_CONFIG[name].color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(v) => [`${v}건`, '단속 건수']} />
                </PieChart>
              </ResponsiveContainer>
              <div className={styles.pieLegend}>
                {typeData.map(({ name, value }) => {
                  const pct = total ? Math.round((value / total) * 100) : 0
                  return (
                    <div key={name} className={styles.pieLegendRow}>
                      <div className={styles.pieLegendLeft}>
                        <span className={styles.pieDot} style={{ backgroundColor: TYPE_CONFIG[name].color }} />
                        <span className={styles.pieLegendName}>{name}</span>
                      </div>
                      <span className={styles.pieLegendVal}>{value}건 ({pct}%)</span>
                    </div>
                  )
                })}
              </div>
            </div>
          </>
        )}

        {/* 시간대별 단속 현황 (전체 너비) */}
        <div className={styles.chartCard}>
          <div className={styles.chartCardHeader}>
            <svg className={styles.chartHeaderIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            시간대별 단속 현황
          </div>
          {violationsLoading ? (
            <div className={styles.status}>불러오는 중...</div>
          ) : violationsFailed ? (
            <div className={styles.statusError}>단속 기록 서버에 연결할 수 없습니다.</div>
          ) : (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={timeData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="헬멧 미착용" stackId="a" fill="#f59e0b" radius={[0, 0, 0, 0]} />
                <Bar dataKey="다인 탑승"   stackId="a" fill="#3b82f6" />
                <Bar dataKey="인도 주행"   stackId="a" fill="#ef4444" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  )
}

export default StatsPage
