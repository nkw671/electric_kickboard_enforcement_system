import { useMemo } from 'react'
import { TYPE_COLOR } from '../constants'
import { DUMMY_VIOLATIONS } from '../data/dummyData'
import useApi from '../hooks/useApi'
import styles from './StatsPage.module.css'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'

const TYPES = ['헬멧 미착용', '다인 탑승', '인도 주행']

function DonutChart({ data, total }) {
  const cx = 55, cy = 55, r = 45, sw = 18
  const circ = 2 * Math.PI * r
  let angle = -90

  return (
    <svg width={110} height={110} style={{ flexShrink: 0 }}>
      <circle cx={cx} cy={cy} r={r} fill="none" stroke="#e4e1d8" strokeWidth={sw} />
      {total > 0 && data.map(({ name, value }) => {
        const pct = value / total
        const dashLen = pct * circ
        const rot = angle
        angle += pct * 360
        return (
          <circle
            key={name}
            cx={cx} cy={cy} r={r}
            fill="none"
            stroke={TYPE_COLOR[name]}
            strokeWidth={sw}
            strokeDasharray={`${dashLen} ${circ}`}
            transform={`rotate(${rot}, ${cx}, ${cy})`}
          />
        )
      })}
    </svg>
  )
}

function StatsPage() {
  const { data: apiData, loading } = useApi('/api/violations?limit=9999')
  const violations = (apiData && apiData.length > 0) ? apiData : DUMMY_VIOLATIONS
  const total = violations.length

  const typeData = useMemo(() =>
    TYPES.map(type => {
      const value = violations.filter(v => v.type === type).length
      return { name: type, value, pct: total ? +((value / total) * 100).toFixed(1) : 0 }
    })
  , [violations, total])

  const timeData = useMemo(() => {
    const buckets = Array.from({ length: 24 }, (_, h) => ({
      time: String(h).padStart(2, '0'),
      '헬멧 미착용': 0, '다인 탑승': 0, '인도 주행': 0,
    }))
    violations.forEach(v => {
      const h = parseInt(v.timestamp?.slice(11, 13) ?? '0')
      if (h >= 0 && h < 24 && v.type in buckets[h]) buckets[h][v.type]++
    })
    return buckets
  }, [violations])

  const locationData = useMemo(() => {
    const counts = {}
    violations.forEach(v => {
      const key = v.location || v.camera
      counts[key] = (counts[key] || 0) + 1
    })
    const sorted = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
    const max = sorted[0]?.[1] || 1
    return sorted.map(([location, count], i) => ({
      location, count,
      pct: total ? +((count / total) * 100).toFixed(1) : 0,
      barW: Math.round((count / max) * 100),
      rank: String(i + 1).padStart(2, '0'),
    }))
  }, [violations, total])

  if (loading) return <div className={styles.status}>불러오는 중...</div>

  return (
    <div className={styles.page}>
      <div className={styles.topbar}>
        <div>
          <div className={styles.crumb}>OPERATIONS / ANALYTICS</div>
          <h2 className={styles.pageTitle}>단속 통계</h2>
        </div>
      </div>

      <div className={styles.content}>
        {/* KPI 4종 */}
        <div className={styles.kpiRow}>
          {/* 전체 카드 (navy) */}
          <div className={`${styles.kpiCard} ${styles.kpiNavy}`}>
            <div className={styles.kpiLabelRow}>
              <span className={styles.kpiLed} style={{ backgroundColor: 'rgba(255,255,255,0.35)' }} />
              <span className={styles.kpiLabelTxt}>전체 단속</span>
            </div>
            <span className={styles.kpiVal} data-numeric>{total}</span>
            <div className={styles.kpiFooter}>
              <span className={styles.kpiDelta}>— vs 전월</span>
            </div>
          </div>

          {typeData.map(({ name, value, pct }) => {
            const dim = value === 0
            return (
              <div key={name} className={styles.kpiCard}>
                <div className={styles.kpiLabelRow}>
                  <span
                    className={styles.kpiLed}
                    style={{ backgroundColor: dim ? 'var(--ink-200)' : TYPE_COLOR[name] }}
                  />
                  <span className={styles.kpiLabelTxt}>{name}</span>
                </div>
                <span
                  className={styles.kpiVal}
                  data-numeric
                  style={{ color: dim ? 'var(--ink-300)' : TYPE_COLOR[name] }}
                >
                  {value}
                </span>
                <div className={styles.kpiFooter}>
                  <span className={styles.kpiPct} data-numeric>{pct}%</span>
                  <span className={styles.kpiDelta}>— vs 전월</span>
                </div>
              </div>
            )
          })}
        </div>

        {/* 도넛 카드 + TOP5 카드 (2열 그리드) */}
        <div className={styles.chartRow}>
          {/* 위반 유형 분포 */}
          <div className={styles.chartCard}>
            <div className={styles.chartCardHeader}>
              <div className={styles.chartCardTitle}>
                위반 유형 분포
                <small>누적 합산</small>
              </div>
            </div>
            <div className={styles.donutSection}>
              <div className={styles.donutWrap}>
                <DonutChart data={typeData} total={total} />
                <div className={styles.donutCenter}>
                  <span className={styles.donutNum} data-numeric>{total}</span>
                  <span className={styles.donutSub}>TOTAL</span>
                </div>
              </div>
              <div className={styles.donutLegend}>
                {typeData.map(({ name, value, pct }) => (
                  <div key={name} className={styles.donutLegendItem}>
                    <span className={styles.donutDot} style={{ backgroundColor: TYPE_COLOR[name] }} />
                    <span className={styles.donutLegendName}>{name}</span>
                    <span className={styles.donutLegendVal} data-numeric>{value}건 · {pct}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 주요 단속 위치 TOP 5 */}
          <div className={styles.chartCard}>
            <div className={styles.chartCardHeader}>
              <div className={styles.chartCardTitle}>
                주요 단속 위치 TOP 5
                <small>누적 기준</small>
              </div>
            </div>
            <div className={styles.top5Section}>
              {locationData.length === 0 ? (
                <div className={styles.emptyState}>
                  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
                  </svg>
                  데이터 없음
                </div>
              ) : locationData.map(({ location, count, pct, barW, rank }) => (
                <div key={location} className={styles.top5Row}>
                  <span className={styles.top5Rank} data-numeric>{rank}</span>
                  <div className={styles.top5Info}>
                    <div className={styles.top5Meta}>
                      <span className={styles.top5Name}>{location}</span>
                      <span className={styles.top5Count} data-numeric>{count}건 · {pct}%</span>
                    </div>
                    <div className={styles.top5Track}>
                      <div className={styles.top5Bar} style={{ width: `${barW}%` }} />
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 시간대별 차트 */}
        <div className={styles.chartCard}>
          <div className={styles.chartCardHeader}>
            <div className={styles.chartCardTitle}>
              시간대별 단속 현황
              <small>0시–23시 시간대별 위반 건수</small>
            </div>
            <div className={styles.chartLegend}>
              {TYPES.map(t => (
                <div key={t} className={styles.legendItem}>
                  <span className={styles.legendSq} style={{ backgroundColor: TYPE_COLOR[t] }} />
                  {t}
                </div>
              ))}
            </div>
          </div>
          <div className={styles.barChartWrap}>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={timeData} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e4e1d8" vertical={false} />
                <XAxis
                  dataKey="time"
                  tick={{ fontFamily: 'IBM Plex Mono', fontSize: 10, fill: '#7a849a' }}
                  axisLine={false}
                  tickLine={false}
                  interval={0}
                  tickFormatter={(v) => parseInt(v) % 6 === 0 ? v : ''}
                />
                <YAxis
                  tick={{ fontFamily: 'IBM Plex Mono', fontSize: 10, fill: '#7a849a' }}
                  axisLine={false}
                  tickLine={false}
                  allowDecimals={false}
                />
                <Tooltip
                  contentStyle={{
                    fontFamily: 'IBM Plex Mono',
                    fontSize: 12,
                    border: '1px solid #e4e1d8',
                    borderRadius: 4,
                    backgroundColor: '#ffffff',
                  }}
                />
                <Bar dataKey="헬멧 미착용" stackId="a" fill={TYPE_COLOR['헬멧 미착용']} />
                <Bar dataKey="다인 탑승"   stackId="a" fill={TYPE_COLOR['다인 탑승']} />
                <Bar dataKey="인도 주행"   stackId="a" fill={TYPE_COLOR['인도 주행']} radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

export default StatsPage
