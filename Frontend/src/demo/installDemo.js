/**
 * 데모 모드에서 fetch/EventSource를 가로채 가짜 응답을 돌려준다.
 * 페이지 코드는 그대로 두고, 실제 서버와 같은 응답 형태(백엔드 PR #39 기준)를 흉내 낸다.
 * - GET  /api/violations  → Spring Page 형태({ content, totalElements, ... }), 최신순
 * - GET  /api/stats       → { overall, hourly, topLocations }
 * - SSE  /api/stream      → 10~20초마다 'violation' 이벤트 (목록에도 추가됨)
 * - GET/POST/DELETE /zones → 메모리에만 저장
 */
import { TYPE_COLOR } from '../constants'

const TYPES = ['헬멧 미착용', '헬멧 미착용', '헬멧 미착용', '인도 주행', '인도 주행', '다인 탑승']
const CAMERAS = [
  { camera: 'CAM-01', location: '정문 앞 인도' },
  { camera: 'CAM-02', location: '학생회관 앞 사거리' },
  { camera: 'CAM-03', location: '도서관 옆 보행로' },
]
const INITIAL_COUNT = 240
const INITIAL_DAYS = 7

// 새로고침해도 기록이 똑같이 나오도록 시드 고정 난수를 쓴다.
let seed = 20261126
function random() {
  seed = (seed * 1664525 + 1013904223) % 4294967296
  return seed / 4294967296
}
const pick = (arr) => arr[Math.floor(random() * arr.length)]

const pad = (n) => String(n).padStart(2, '0')
function formatTimestamp(d) {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
}

// 상세 화면에 보여줄 단속 이미지. 실제 캡처 대신 위반 유형이 표시된 간단한 그림.
// 박스 색은 화면의 위반 유형 색(constants.js)을 그대로 쓴다.
function snapshotUrl(type, camera, timestamp) {
  const color = TYPE_COLOR[type]
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="800" height="450" viewBox="0 0 800 450">
<rect width="800" height="450" fill="#334155"/>
<polygon points="0,450 0,300 260,220 800,260 800,450" fill="#475569"/>
<line x1="0" y1="330" x2="800" y2="300" stroke="#94a3b8" stroke-width="3" stroke-dasharray="30 20"/>
<g transform="translate(330 150)">
<circle cx="40" cy="20" r="16" fill="#cbd5e1"/><rect x="26" y="38" width="28" height="70" rx="10" fill="#cbd5e1"/>
<line x1="20" y1="170" x2="110" y2="170" stroke="#e2e8f0" stroke-width="6"/>
<line x1="95" y1="170" x2="85" y2="80" stroke="#e2e8f0" stroke-width="5"/>
<circle cx="20" cy="178" r="10" fill="#e2e8f0"/><circle cx="110" cy="178" r="10" fill="#e2e8f0"/>
</g>
<rect x="310" y="130" width="170" height="215" fill="none" stroke="${color}" stroke-width="4"/>
<rect x="310" y="100" width="170" height="30" fill="${color}"/>
<text x="320" y="122" font-family="sans-serif" font-size="18" font-weight="bold" fill="#fff">${type}</text>
<text x="60" y="400" font-family="monospace" font-size="18" fill="#f8fafc">${camera}  ${timestamp}</text>
<text x="740" y="400" font-family="sans-serif" font-size="14" fill="#cbd5e1" text-anchor="end">데모 이미지</text>
</svg>`
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

let nextId = 1
function makeViolation(date) {
  const type = pick(TYPES)
  const { camera, location } = pick(CAMERAS)
  const timestamp = formatTimestamp(date)
  return {
    id: nextId++,
    type,
    camera,
    location,
    timestamp,
    confidence: 70 + Math.floor(random() * 29),
    image_url: snapshotUrl(type, camera, timestamp),
  }
}

// 최근 7일, 출퇴근·점심 시간대에 몰리게 분포시킨다.
const BUSY_HOURS = [8, 8, 9, 9, 12, 12, 13, 17, 18, 18, 19, 21]
const violations = (() => {
  const now = Date.now()
  const list = []
  for (let i = 0; i < INITIAL_COUNT; i++) {
    const d = new Date(now - Math.floor(random() * INITIAL_DAYS) * 86400000)
    d.setHours(random() < 0.75 ? pick(BUSY_HOURS) : Math.floor(random() * 24),
      Math.floor(random() * 60), Math.floor(random() * 60))
    if (d.getTime() > now) d.setTime(now - Math.floor(random() * 3600000))
    list.push(d)
  }
  list.sort((a, b) => a - b)
  return list.map(makeViolation).reverse() // 최신순
})()

let zones = [
  { name: 'Zone-1', pts: [[0, 300], [260, 220], [330, 225], [120, 450], [0, 450]], color: [0, 0, 255] },
]

function pageResponse(params) {
  const type = params.get('type')
  const list = type ? violations.filter(v => v.type === type) : violations
  const size = Number(params.get('size') || params.get('limit') || 20)
  const page = Number(params.get('page') || 0)
  return {
    content: list.slice(page * size, (page + 1) * size),
    totalElements: list.length,
    totalPages: Math.ceil(list.length / size),
    number: page,
    size,
  }
}

function statsResponse() {
  const count = (type) => violations.filter(v => v.type === type).length
  const hourly = Array.from({ length: 24 }, (_, hour) => ({ hour, count: 0 }))
  const byLocation = {}
  violations.forEach(v => {
    hourly[Number(v.timestamp.slice(11, 13))].count++
    byLocation[v.location] = (byLocation[v.location] || 0) + 1
  })
  return {
    overall: {
      total: violations.length,
      helmet: count('헬멧 미착용'),
      sidewalk: count('인도 주행'),
      multiRider: count('다인 탑승'),
    },
    hourly,
    topLocations: Object.entries(byLocation)
      .map(([location, n]) => ({ location, count: n }))
      .sort((a, b) => b.count - a.count),
  }
}

const json = (body) => new Response(JSON.stringify(body), {
  status: 200,
  headers: { 'Content-Type': 'application/json' },
})
const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const realFetch = window.fetch.bind(window)
window.fetch = async (input, init = {}) => {
  const url = new URL(typeof input === 'string' ? input : input.url, window.location.href)
  const method = (init.method || 'GET').toUpperCase()
  const path = url.pathname

  if (path.endsWith('/api/violations')) {
    await delay(150)
    return json(pageResponse(url.searchParams))
  }
  if (path.endsWith('/api/stats')) {
    await delay(150)
    return json(statsResponse())
  }
  if (path.endsWith('/zones')) {
    await delay(150)
    if (method === 'POST') {
      zones = JSON.parse(init.body).zones || []
      return json({ saved: zones.length })
    }
    if (method === 'DELETE') {
      zones = []
      return json({ cleared: true })
    }
    return json({ zones })
  }
  return realFetch(input, init)
}

// useSSE가 쓰는 부분(onopen, onerror, addEventListener, close)만 흉내 낸다.
class DemoEventSource extends EventTarget {
  constructor(url) {
    super()
    this.url = url
    this.onopen = null
    this.onerror = null
    this.timer = setTimeout(() => {
      this.onopen?.(new Event('open'))
      this.scheduleNext()
    }, 300)
  }

  scheduleNext() {
    this.timer = setTimeout(() => {
      const v = makeViolation(new Date())
      violations.unshift(v)
      this.dispatchEvent(new MessageEvent('violation', { data: JSON.stringify(v) }))
      this.scheduleNext()
    }, 10000 + random() * 10000)
  }

  close() {
    clearTimeout(this.timer)
  }
}

const RealEventSource = window.EventSource
window.EventSource = function (url, options) {
  if (new URL(url, window.location.href).pathname.endsWith('/api/stream')) {
    return new DemoEventSource(url)
  }
  return new RealEventSource(url, options)
}
