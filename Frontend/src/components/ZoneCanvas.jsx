import { useRef, useEffect, useState } from 'react'
import styles from './ZoneCanvas.module.css'

// 클래스명        : ZoneCanvas
// 기능           : 비디오 스트림 위에 오버레이되는 Zone 그리기 캔버스 컴포넌트
// 내장 함수 목록  : handleClick()    - 클릭 시 꼭짓점 추가 또는 Zone 완성
//                  handleMouseMove() - 마우스 이동 시 미리보기 업데이트
//                  finishZone()      - 현재 그리는 Zone 완성 및 서버 저장
//                  clearZones()      - 전체 Zone 초기화

const CLOSE_RADIUS = 15                                                    // 첫 꼭짓점 클릭 인식 반경(px, 영상 좌표)
const ZONE_COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFD166']
const API_BASE = 'http://localhost:8000'

function ZoneCanvas({ imgRef }) {
  const canvasRef = useRef(null)
  const [zones, setZones] = useState([])
  const [currentPts, setCurrentPts] = useState([])                        // 그리는 중인 Zone 꼭짓점 (영상 좌표)
  const [isDrawing, setIsDrawing] = useState(false)
  const [mousePos, setMousePos] = useState(null)
  const [videoDims, setVideoDims] = useState({ w: 0, h: 0 })

  // img naturalWidth/Height 를 폴링하여 캔버스 버퍼를 영상 해상도로 맞춤
  useEffect(() => {
    const img = imgRef.current
    if (!img) return

    function trySync() {
      if (img.naturalWidth > 0) {
        const canvas = canvasRef.current
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        setVideoDims({ w: img.naturalWidth, h: img.naturalHeight })
        return true
      }
      return false
    }

    if (trySync()) return
    const timer = setInterval(() => { if (trySync()) clearInterval(timer) }, 200)
    return () => clearInterval(timer)
  }, [imgRef])

  // 초기 Zone 불러오기
  useEffect(() => {
    fetch(`${API_BASE}/zones`)
      .then(r => r.json())
      .then(data => setZones(data.zones || []))
      .catch(() => {})
  }, [])

  // 캔버스 렌더링
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || videoDims.w === 0) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, videoDims.w, videoDims.h)

    // 저장된 Zone 렌더링
    zones.forEach((zone, i) => {
      drawPoly(ctx, zone.pts, ZONE_COLORS[i % ZONE_COLORS.length], true)
    })

    // 그리는 중인 Zone 미리보기
    if (currentPts.length > 0) {
      const preview = mousePos ? [...currentPts, mousePos] : currentPts
      drawPoly(ctx, preview, ZONE_COLORS[zones.length % ZONE_COLORS.length], false)

      // 첫 꼭짓점 닫기 표시 (3점 이상일 때)
      if (currentPts.length >= 3) {
        ctx.beginPath()
        ctx.arc(currentPts[0][0], currentPts[0][1], CLOSE_RADIUS, 0, Math.PI * 2)
        ctx.strokeStyle = 'rgba(255,255,255,0.85)'
        ctx.lineWidth = 2
        ctx.stroke()
      }
    }
  }, [zones, currentPts, mousePos, videoDims])

  // 함수 이름 : drawPoly()
  // 기능      : 주어진 좌표 배열로 다각형을 캔버스에 그림
  // 파라미터  : CanvasRenderingContext2D ctx, number[][] pts, string color, boolean closed
  // 반환값    : 없음
  function drawPoly(ctx, pts, color, closed) {
    if (pts.length === 0) return

    pts.forEach(([x, y]) => {
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
    })

    if (pts.length < 2) return

    ctx.beginPath()
    ctx.moveTo(pts[0][0], pts[0][1])
    pts.slice(1).forEach(([x, y]) => ctx.lineTo(x, y))
    if (closed) {
      ctx.closePath()
      ctx.fillStyle = color + '40'                                         // 25% 불투명도
      ctx.fill()
    }
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.stroke()
  }

  // 함수 이름 : getVideoPos()
  // 기능      : 마우스 이벤트의 CSS 좌표를 영상 해상도 기준 좌표로 변환
  // 파라미터  : MouseEvent e
  // 반환값    : [number, number] 영상 좌표
  function getVideoPos(e) {
    const rect = canvasRef.current.getBoundingClientRect()
    const x = (e.clientX - rect.left) / rect.width * videoDims.w
    const y = (e.clientY - rect.top) / rect.height * videoDims.h
    return [x, y]
  }

  // 함수 이름 : handleClick()
  // 기능      : 클릭 시 꼭짓점 추가, 첫 꼭짓점 근처 클릭 시 Zone 완성
  // 파라미터  : MouseEvent e
  // 반환값    : 없음
  function handleClick(e) {
    if (!isDrawing) return
    const pos = getVideoPos(e)

    if (currentPts.length >= 3) {
      const dist = Math.hypot(pos[0] - currentPts[0][0], pos[1] - currentPts[0][1])
      if (dist < CLOSE_RADIUS) {
        finishZone()
        return
      }
    }
    setCurrentPts(prev => [...prev, pos])
  }

  function handleMouseMove(e) {
    if (!isDrawing || currentPts.length === 0) return
    setMousePos(getVideoPos(e))
  }

  // 함수 이름 : finishZone()
  // 기능      : 현재 그리는 Zone을 완성하고 초기 상태로 복귀, 서버 저장은 비동기로 처리
  // 파라미터  : 없음
  // 반환값    : 없음
  function finishZone() {
    if (currentPts.length < 3) return
    const pts = currentPts.map(([x, y]) => [Math.round(x), Math.round(y)])
    const newZone = { name: `Zone ${zones.length + 1}`, pts }
    const updated = [...zones, newZone]
    setZones(updated)
    setCurrentPts([])
    setMousePos(null)
    postZones(updated)
  }

  // 함수 이름 : postZones()
  // 기능      : Zone 목록을 서버에 저장 (POST /zones), fire-and-forget
  // 파라미터  : object[] updated -> 저장할 Zone 목록
  // 반환값    : 없음
  async function postZones(updated) {
    try {
      await fetch(`${API_BASE}/zones`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ zones: updated.map(z => ({ name: z.name, pts: z.pts })) }),
      })
    } catch (err) {
      console.error('Zone 저장 실패:', err)
    }
  }

  function undoLastPt() {
    setCurrentPts(prev => prev.slice(0, -1))
  }

  // 함수 이름 : clearZones()
  // 기능      : 서버 및 로컬 Zone 전체 초기화 (DELETE /zones)
  // 파라미터  : 없음
  // 반환값    : Promise<void>
  async function clearZones() {
    try {
      await fetch(`${API_BASE}/zones`, { method: 'DELETE' })
      setZones([])
      setCurrentPts([])
    } catch (err) {
      console.error('Zone 초기화 실패:', err)
    }
  }

  return (
    <>
      <canvas
        ref={canvasRef}
        className={styles.canvas}
        style={{ pointerEvents: isDrawing ? 'auto' : 'none', cursor: 'crosshair' }}
        onClick={handleClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setMousePos(null)}
      />
      <div className={styles.controls}>
        {isDrawing ? (
          <>
            <button
              className={styles.btn}
              onClick={() => { setCurrentPts([]); setMousePos(null); setIsDrawing(false) }}
            >
              그리기 완료
            </button>
            <button
              className={styles.btn}
              onClick={undoLastPt}
              disabled={currentPts.length === 0}
            >
              되돌리기
            </button>
          </>
        ) : (
          <>
            <button className={styles.btn} onClick={() => setIsDrawing(true)}>
              Zone 그리기
            </button>
            {zones.length > 0 && (
              <button className={`${styles.btn} ${styles.btnDanger}`} onClick={clearZones}>
                구역 초기화
              </button>
            )}
          </>
        )}
      </div>
    </>
  )
}

export default ZoneCanvas
