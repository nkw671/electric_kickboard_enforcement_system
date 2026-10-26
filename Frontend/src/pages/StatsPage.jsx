import { useMemo, useState } from 'react'
import useApi from '../hooks/useApi'
import { apiUrl } from '../config'
import { toViolationList } from '../utils'
import { TYPE_ICON, TOTAL_ICON } from '../typeIcons'
import styles from './StatsPage.module.css'
import {
  PieChart, Pie, Cell, Sector, Tooltip, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts'

// color: 아이콘·글씨용 진한 색 / chart: 차트용 연한 색(아이콘 배경과 같은 파스텔 계열, 300 단계)
// icon / tint: 요약 카드 아이콘(600 단계)과 아이콘 배경(100 단계, 70% 불투명)
const TYPE_CONFIG = {
  '헬멧 미착용': { color: '#f59e0b', chart: '#fcd34d', icon: '#d97706', tint: 'rgba(254, 243, 199, 0.7)', borderColor: '#fde68a' },
  '다인 탑승':   { color: '#8b5cf6', chart: '#c4b5fd', icon: '#7c3aed', tint: 'rgba(237, 233, 254, 0.7)', borderColor: '#ddd6fe' },
  '인도 주행':   { color: '#ef4444', chart: '#fca5a5', icon: '#dc2626', tint: 'rgba(254, 226, 226, 0.7)', borderColor: '#fecaca' },
}

// 시간대별 막대를 쌓는 순서(아래 → 위)
const STACK_ORDER = ['헬멧 미착용', '다인 탑승', '인도 주행']
// X축은 3시간 간격(00:00, 03:00 ... 21:00)만 표시
const X_TICKS = Array.from({ length: 8 }, (_, i) => `${String(i * 3).padStart(2, '0')}:00`)

// 누적 막대 조각. 그 시간대에서 맨 위에 보이는 조각만 위쪽 모서리를 둥글게 하고, 나머지는 직각.
// (인도 주행이 0인 시간대는 다인 탑승이 맨 위가 되므로 Bar마다 radius를 고정하면 안 됨)
function stackSegment(dataKey) {
  const above = STACK_ORDER.slice(STACK_ORDER.indexOf(dataKey) + 1)
  return function StackSegment({ x, y, width, height, fill, payload }) {
    if (!height || height <= 0) return null
    const r = above.every(k => !payload[k]) ? Math.min(4, height, width / 2) : 0
    const d = `M${x},${y + height} V${y + r} Q${x},${y} ${x + r},${y} H${x + width - r} ` +
      `Q${x + width},${y} ${x + width},${y + r} V${y + height} Z`
    return <path d={d} fill={fill} />
  }
}
const SEGMENT_SHAPE = Object.fromEntries(STACK_ORDER.map(k => [k, stackSegment(k)]))


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
  // 인도 주행 많은 구역 TOP 5 (백엔드 #39의 /api/stats topLocations, 이전 응답 형태에는 없음).
  // AI는 인도 주행에만 location(구역 이름)을 넣고 나머지 위반은 ""로 보내서, 이름 없는 항목은 뺀다.
  // → 사실상 인도 주행이 많이 감지된 구역 순위
  const topLocations = (statsData?.topLocations ?? []).filter(l => l.location?.trim()).slice(0, 5)
  const topMax = topLocations[0]?.count || 1

  // 폴링 응답은 값이 같아도 매번 새 객체라서, overall을 의존성으로 두면 3초마다
  // 차트 애니메이션이 처음부터 다시 재생된다. 값이 바뀔 때만 새 배열을 만든다.
  const helmet = overall?.helmet ?? 0
  const multiRider = overall?.multiRider ?? 0
  const sidewalk = overall?.sidewalk ?? 0
  // 도넛 조각에 마우스를 올리면 가운데에 그 유형의 건수를 보여준다(툴팁 대신 — 가운데 글씨와 겹치지 않게)
  const [hoverType, setHoverType] = useState(null)

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
      const hour = String(h).padStart(2, '0') + ':00'
      buckets[hour] = { time: hour, '헬멧 미착용': 0, '다인 탑승': 0, '인도 주행': 0 }
    }
    violations.forEach(v => {
      const hour = v.timestamp.slice(11, 13) + ':00'
      if (v.type in (buckets[hour] || {})) buckets[hour][v.type]++
    })
    return Object.values(buckets)
  }, [violations])
  // 위와 같은 이유로, 집계 결과가 실제로 바뀔 때만 차트 데이터를 교체한다.
  const timeDataKey = JSON.stringify(rawTimeData)
  const timeData = useMemo(() => JSON.parse(timeDataKey), [timeDataKey])

  // Y축은 기본 0~40(10 간격). 한 시간대 합계가 40을 넘으면 10 단위(60 초과 시 20 단위)로 늘림.
  const { yMax, yTicks } = useMemo(() => {
    const peak = Math.max(0, ...timeData.map(d => STACK_ORDER.reduce((sum, k) => sum + d[k], 0)))
    const step = peak <= 60 ? 10 : 20
    const max = Math.max(40, Math.ceil(peak / step) * step)
    return { yMax: max, yTicks: Array.from({ length: max / step + 1 }, (_, i) => i * step) }
  }, [timeData])

  if (statsLoading && violationsLoading) return <div className={styles.status}>불러오는 중...</div>

  return (
    <div className={styles.page}>
      {/* 헤더 */}
      <div className={styles.header}>
        <h2 className={styles.title}>단속 통계</h2>
        <p className={styles.desc}>
          누적 단속 기록을 위반 유형 및 시간대별로 분석합니다.
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
                  <span className={styles.summaryIconBox}>{TOTAL_ICON}</span>
                  전체 단속
                </div>
                <p className={styles.summaryVal}>{total}<span className={styles.summaryUnit}>건</span></p>
              </div>
              {typeData.map(({ name, value }) => {
                const cfg = TYPE_CONFIG[name]
                const pct = total ? Math.round((value / total) * 100) : 0
                return (
                  <div key={name} className={styles.summaryCard}>
                    <div className={styles.summaryCardLabel}>
                      <span className={styles.summaryIconBox} style={{ backgroundColor: cfg.tint, color: cfg.icon }}>{TYPE_ICON[name]}</span>
                      {name}
                    </div>
                    <p className={styles.summaryVal}>{value}<span className={styles.summaryUnit}>건 ({pct}%)</span></p>
                  </div>
                )
              })}
            </div>

            {/* 위반 유형 분포(7) + 인도 주행 많은 구역 TOP 5(5), 두 카드 높이 맞춤 */}
            <div className={styles.chartRow}>
            <div className={`${styles.chartCard} ${styles.pieCard}`}>
              <div className={styles.chartCardHeader}>
                <svg className={styles.chartHeaderIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
                위반 유형 분포
                <span className={styles.chartCardMeta}>전체 {total}건 기준</span>
              </div>
              <div className={styles.pieBody}>
              <div className={styles.pieWrap}>
                <PieChart width={160} height={160}>
                  {/* 애니메이션을 끔: recharts Pie는 다시 렌더링될 때마다(3초 폴링 등) 애니메이션을
                      처음부터 다시 시작해서, 타이밍에 따라 몇 초 동안 빈 차트로 보이는 문제가 있었음 */}
                  <Pie
                    data={typeData} cx="50%" cy="50%" innerRadius={50} outerRadius={72} paddingAngle={3} dataKey="value"
                    isAnimationActive={false}
                    onMouseEnter={(_, i) => setHoverType(typeData[i]?.name ?? null)}
                    onMouseLeave={() => setHoverType(null)}
                    // 마우스를 올린 조각은 바깥으로 6px 키우고, 나머지 조각은 흐리게
                    shape={(props) => {
                      const name = props.payload?.name
                      const active = hoverType === name
                      return (
                        <Sector
                          {...props}
                          outerRadius={props.outerRadius + (active ? 6 : 0)}
                          fillOpacity={hoverType && !active ? 0.35 : 1}
                          style={{ transition: 'fill-opacity 0.15s', cursor: 'default' }}
                        />
                      )
                    }}
                  >
                    {typeData.map(({ name }) => (
                      <Cell key={name} fill={TYPE_CONFIG[name].chart} />
                    ))}
                  </Pie>
                </PieChart>
                {/* 도넛 가운데 총 건수 */}
                <div className={styles.pieCenter}>
                  <span className={styles.pieCenterVal} style={hoverType ? { color: TYPE_CONFIG[hoverType].icon } : undefined}>
                    {hoverType ? typeData.find(t => t.name === hoverType).value : total}
                  </span>
                  <span className={styles.pieCenterLabel}>{hoverType ?? '총 단속'}</span>
                </div>
              </div>
              <div className={styles.pieLegend}>
                {typeData.map(({ name, value }) => {
                  const pct = total ? Math.round((value / total) * 100) : 0
                  return (
                    <div key={name} className={styles.pieLegendRow}>
                      <div className={styles.pieLegendLeft}>
                        <span className={styles.pieDot} style={{ backgroundColor: TYPE_CONFIG[name].chart }} />
                        <span className={styles.pieLegendName}>{name}</span>
                      </div>
                      <span className={styles.pieLegendVal}>{value}건 ({pct}%)</span>
                    </div>
                  )
                })}
              </div>
              </div>
            </div>

            <div className={`${styles.chartCard} ${styles.topCard}`}>
              <div className={styles.chartCardHeader}>
                {/* Lucide: map-pin */}
                <svg className={styles.chartHeaderIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M20 10c0 4.993-5.539 10.193-7.399 11.799a1 1 0 0 1-1.202 0C9.539 20.193 4 14.993 4 10a8 8 0 0 1 16 0"/>
                  <circle cx="12" cy="10" r="3"/>
                </svg>
                인도 주행 많은 구역 TOP 5
              </div>
              {topLocations.length === 0 ? (
                <div className={styles.topEmpty}>인도 주행이 감지된 구역이 없습니다.</div>
              ) : (
                <ol className={styles.topList}>
                  {/* 동률은 같은 순위(1, 1, 3 …)로 표시하고, 최다 건수 항목은 모두 파란 막대 */}
                  {topLocations.map(({ location, count }) => {
                    const rank = 1 + topLocations.filter(l => l.count > count).length
                    return (
                      <li key={location} className={styles.topItem}>
                        <div className={styles.topRow}>
                          <span className={`${styles.topRank} ${rank <= 3 ? styles.topRankHigh : ''}`}>{rank}</span>
                          <span className={styles.topName}>{location}</span>
                          <span className={styles.topCount}>{count}<span className={styles.topUnit}>건</span></span>
                        </div>
                        {/* 1위 대비 비율 막대 */}
                        <div className={styles.topBarTrack}>
                          <div
                            className={`${styles.topBar} ${count === topMax ? styles.topBarFirst : ''}`}
                            style={{ width: `${Math.round((count / topMax) * 100)}%` }}
                          />
                        </div>
                      </li>
                    )
                  })}
                </ol>
              )}
            </div>
            </div>
          </>
        )}

        {/* 시간대별 단속 현황 (전체 너비) */}
        <div className={styles.chartCard}>
          <div className={styles.chartCardHeader}>
            <svg className={styles.chartHeaderIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              {/* Lucide: clock */}
              <circle cx="12" cy="12" r="10"/>
              <polyline points="12 6 12 12 16 14"/>
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
                <CartesianGrid vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="time" ticks={X_TICKS} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
                <YAxis domain={[0, yMax]} ticks={yTicks} tick={{ fontSize: 11, fill: '#94a3b8' }} axisLine={false} tickLine={false} allowDecimals={false} />
                <Tooltip cursor={{ fill: '#f8fafc' }} labelFormatter={(t) => `${t} ~ ${t.slice(0, 2)}:59`} />
                {/* 범례 기본 정렬(이름순) 대신 위 요약 카드와 같은 순서로 */}
                <Legend wrapperStyle={{ fontSize: 12 }} iconType="circle" iconSize={8}
                  itemSorter={(item) => typeData.findIndex(t => t.name === item.value)}
                  formatter={(value) => <span style={{ color: '#475569' }}>{value}</span>} />
                {STACK_ORDER.map(k => (
                  <Bar key={k} dataKey={k} stackId="a" fill={TYPE_CONFIG[k].chart} shape={SEGMENT_SHAPE[k]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  )
}

export default StatsPage
