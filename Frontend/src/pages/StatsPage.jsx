import { useMemo } from 'react'
import useApi from '../hooks/useApi'
import styles from './StatsPage.module.css'
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts'

const TYPES = ['헬멧 미착용', '다인 탑승', '인도 주행']

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
  '헬멧 미착용': { color: '#0ea5e9', borderColor: '#bae6fd' },
  '다인 탑승':   { color: '#f59e0b', borderColor: '#fde68a' },
  '인도 주행':   { color: '#ec4899', borderColor: '#fbcfe8' },
}


function StatsPage() {
  const { data: apiData, loading, error } = useApi('/api/violations?limit=9999')
  const violations = useMemo(() => apiData || [], [apiData])

  const total = violations.length

  const typeData = useMemo(() =>
    TYPES.map(type => ({
      name: type,
      value: violations.filter(v => v.type === type).length,
    }))
  , [violations])

  const timeData = useMemo(() => {
    const buckets = {}
    violations.forEach(v => {
      const hour = v.timestamp.slice(11, 13) + '시'
      if (!buckets[hour]) buckets[hour] = { time: hour, '헬멧 미착용': 0, '다인 탑승': 0, '인도 주행': 0 }
      if (v.type in buckets[hour]) buckets[hour][v.type]++
    })
    return Object.values(buckets).sort((a, b) => a.time.localeCompare(b.time))
  }, [violations])

  if (loading) return <div className={styles.status}>불러오는 중...</div>
  if (error) return <div className={styles.statusError}>서버에 연결할 수 없습니다.</div>

  return (
    <div className={styles.page}>
      {/* 헤더 */}
      <div className={styles.header}>
        <h2 className={styles.title}>단속 통계</h2>
      </div>

      <div className={styles.content}>
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
              <Pie data={typeData} cx="50%" cy="50%" innerRadius={34} outerRadius={55} paddingAngle={3} dataKey="value">
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

        {/* 시간대별 단속 현황 (전체 너비) */}
        <div className={styles.chartCard}>
          <div className={styles.chartCardHeader}>
            <svg className={styles.chartHeaderIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            시간대별 단속 현황
          </div>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={timeData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="time" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="헬멧 미착용" stackId="a" fill="#0ea5e9" radius={[0, 0, 0, 0]} />
              <Bar dataKey="다인 탑승"   stackId="a" fill="#f59e0b" />
              <Bar dataKey="인도 주행"   stackId="a" fill="#ec4899" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}

export default StatsPage
