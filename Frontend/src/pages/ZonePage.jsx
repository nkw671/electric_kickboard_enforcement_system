import { useState, useEffect, useRef } from 'react'
import styles from './ZonePage.module.css'

// 스트림이 로드되기 전(미연결 등) 사용할 기본 해상도. 연결되면 영상 실제 해상도로 자동 교체된다.
const DEFAULT_W = 800
const DEFAULT_H = 450

const ZONE_COLORS_BGR = [
  [0,   0,   255],
  [0,   165, 255],
  [0,   255, 0  ],
  [255, 100, 0  ],
  [255, 0,   200],
]

function bgrToRgb([b, g, r]) {
  return `rgb(${r},${g},${b})`
}

function bgrToHex([b, g, r]) {
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`
}

function ZonePage() {
  const canvasRef = useRef(null)
  const imgRef = useRef(null)
  const [zones, setZones] = useState([])
  const [currentPts, setCurrentPts] = useState([])
  const [isDrawing, setIsDrawing] = useState(false)
  const [status, setStatus] = useState(null)
  const [aiConnected, setAiConnected] = useState(false)
  const [streamError, setStreamError] = useState(false)
  const [videoDims, setVideoDims] = useState({ w: DEFAULT_W, h: DEFAULT_H })

  useEffect(() => {
    fetch('/ai/zones')
      .then(r => r.json())
      .then(data => {
        setZones(data.zones || [])
        setAiConnected(true)
      })
      .catch(() => setAiConnected(false))
  }, [])

  // 영상 스트림의 실제 해상도(naturalWidth/Height)를 폴링하여 좌표계를 영상에 맞춘다.
  // AI(detection.py INFER_SIZE)가 어떤 해상도를 쓰든 자동으로 일치하므로 하드코딩이 불필요하다.
  useEffect(() => {
    if (!aiConnected || streamError) return
    const img = imgRef.current
    if (!img) return

    const trySync = () => {
      if (img.naturalWidth > 0) {
        setVideoDims({ w: img.naturalWidth, h: img.naturalHeight })
        return true
      }
      return false
    }

    if (trySync()) return
    const timer = setInterval(() => { if (trySync()) clearInterval(timer) }, 200)
    return () => clearInterval(timer)
  }, [aiConnected, streamError])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, videoDims.w, videoDims.h)

    // 저장된 존 렌더링
    zones.forEach(zone => {
      if (zone.pts.length < 2) return
      const [b, g, r] = zone.color
      const colorRgb = `rgb(${r},${g},${b})`

      ctx.beginPath()
      ctx.moveTo(zone.pts[0][0], zone.pts[0][1])
      zone.pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
      ctx.closePath()
      ctx.fillStyle = `rgba(${r},${g},${b},0.25)`
      ctx.fill()
      ctx.strokeStyle = colorRgb
      ctx.lineWidth = 2
      ctx.stroke()

      zone.pts.forEach(([x, y]) => {
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, Math.PI * 2)
        ctx.fillStyle = colorRgb
        ctx.fill()
      })

      const cx = zone.pts.reduce((s, p) => s + p[0], 0) / zone.pts.length
      const cy = zone.pts.reduce((s, p) => s + p[1], 0) / zone.pts.length
      ctx.fillStyle = colorRgb
      ctx.font = 'bold 14px sans-serif'
      ctx.fillText(zone.name, cx - 20, cy + 5)
    })

    // 현재 그리는 중인 폴리곤
    if (currentPts.length > 0) {
      const color = ZONE_COLORS_BGR[zones.length % ZONE_COLORS_BGR.length]
      const colorRgb = bgrToRgb(color)

      ctx.beginPath()
      ctx.moveTo(currentPts[0][0], currentPts[0][1])
      currentPts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
      ctx.strokeStyle = colorRgb
      ctx.lineWidth = 2
      ctx.setLineDash([6, 4])
      ctx.stroke()
      ctx.setLineDash([])

      currentPts.forEach(([x, y], i) => {
        ctx.beginPath()
        ctx.arc(x, y, i === 0 ? 6 : 4, 0, Math.PI * 2)
        ctx.fillStyle = i === 0 ? '#ffffff' : colorRgb
        ctx.fill()
        ctx.strokeStyle = colorRgb
        ctx.lineWidth = 2
        ctx.stroke()
      })
    }
  }, [zones, currentPts, videoDims])

  const handleCanvasClick = (e) => {
    if (!isDrawing) return
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const x = Math.round((e.clientX - rect.left) * (videoDims.w / rect.width))
    const y = Math.round((e.clientY - rect.top) * (videoDims.h / rect.height))
    setCurrentPts(prev => [...prev, [x, y]])
  }

  const finishZone = () => {
    if (currentPts.length < 3) return
    const color = ZONE_COLORS_BGR[zones.length % ZONE_COLORS_BGR.length]
    const name = `Zone-${zones.length + 1}`
    setZones(prev => [...prev, { name, pts: currentPts, color }])
    setCurrentPts([])
    setIsDrawing(false)
  }

  const cancelDraw = () => {
    setCurrentPts([])
    setIsDrawing(false)
  }

  const deleteZone = (idx) => {
    setZones(prev => prev.filter((_, i) => i !== idx).map((z, i) => ({ ...z, name: `Zone-${i + 1}` })))
  }

  const saveZones = async () => {
    try {
      const res = await fetch('/ai/zones', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ zones: zones.map(z => ({ name: z.name, pts: z.pts, color: z.color })) }),
      })
      if (!res.ok) throw new Error()
      setStatus({ ok: true, msg: '저장 완료' })
    } catch {
      setStatus({ ok: false, msg: '저장 실패 — AI 서버 연결 확인' })
    }
    setTimeout(() => setStatus(null), 3000)
  }

  const deleteAllZones = async () => {
    try {
      await fetch('/ai/zones', { method: 'DELETE' })
      setZones([])
      setStatus({ ok: true, msg: '전체 삭제 완료' })
    } catch {
      setStatus({ ok: false, msg: '삭제 실패 — AI 서버 연결 확인' })
    }
    setTimeout(() => setStatus(null), 3000)
  }

  return (
    <div className={styles.page}>
      <h2 className={styles.title}>구역 설정</h2>

      <div className={styles.layout}>
        {/* 캔버스 영역 */}
        <div className={styles.canvasWrap}>
          <div className={styles.canvasHeader}>
            <span className={`${styles.connDot} ${aiConnected ? styles.connOn : styles.connOff}`} />
            <span className={styles.connLabel}>{aiConnected ? 'AI 서버 연결됨' : 'AI 서버 미연결'}</span>
            {isDrawing && <span className={styles.drawingBadge}>그리는 중 — 클릭으로 꼭짓점 추가</span>}
          </div>
          <div className={styles.canvasStage}>
            {aiConnected && !streamError ? (
              <img
                ref={imgRef}
                src="/ai/video/stream"
                className={styles.canvasVideo}
                alt="stream"
                onError={() => setStreamError(true)}
              />
            ) : (
              <div className={styles.canvasPlaceholder}>
                AI 서버 연결 후 활성화
                <small>(http://localhost:8000)</small>
              </div>
            )}
            <canvas
              ref={canvasRef}
              width={videoDims.w}
              height={videoDims.h}
              className={`${styles.canvas} ${isDrawing ? styles.canvasDrawing : ''}`}
              onClick={handleCanvasClick}
            />
          </div>
          <div className={styles.canvasFooter}>
            {!isDrawing ? (
              <button className={styles.btnPrimary} onClick={() => setIsDrawing(true)}>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <polygon points="3 11 22 2 13 21 11 13 3 11"/>
                </svg>
                그리기 시작
              </button>
            ) : (
              <>
                <button className={styles.btnSuccess} onClick={finishZone} disabled={currentPts.length < 3}>
                  완료 ({currentPts.length}점)
                </button>
                <button className={styles.btnGhost} onClick={cancelDraw}>취소</button>
              </>
            )}
            <span className={styles.hint}>최소 3개 꼭짓점 필요</span>
          </div>
        </div>

        {/* 사이드 패널 */}
        <div className={styles.panel}>
          <div className={styles.panelHeader}>
            <span className={styles.panelTitle}>등록된 구역</span>
            <span className={styles.zoneCnt}>{zones.length}개</span>
          </div>

          <div className={styles.zoneList}>
            {zones.length === 0 ? (
              <div className={styles.emptyZone}>등록된 구역이 없습니다</div>
            ) : (
              zones.map((zone, i) => (
                <div key={i} className={styles.zoneItem}>
                  <span className={styles.zoneDot} style={{ backgroundColor: bgrToHex(zone.color) }} />
                  <span className={styles.zoneName}>{zone.name}</span>
                  <span className={styles.zonePts}>{zone.pts.length}pt</span>
                  <button className={styles.zoneDeleteBtn} onClick={() => deleteZone(i)}>✕</button>
                </div>
              ))
            )}
          </div>

          <div className={styles.panelFooter}>
            {status && (
              <div className={`${styles.statusMsg} ${status.ok ? styles.statusOk : styles.statusErr}`}>
                {status.msg}
              </div>
            )}
            <button className={styles.btnSave} onClick={saveZones} disabled={zones.length === 0}>
              AI 서버에 저장
            </button>
            <button className={styles.btnDelete} onClick={deleteAllZones} disabled={zones.length === 0}>
              전체 삭제
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ZonePage
