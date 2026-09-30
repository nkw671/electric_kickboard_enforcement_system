import { Component } from 'react'
import styles from './ErrorBoundary.module.css'

// 하위 트리에서 렌더링 중 에러가 나면 흰 화면 대신 안내 화면을 보여준다.
// (클래스 컴포넌트인 이유: 에러 캐치는 훅으로 구현할 수 없고 React가
// componentDidCatch/getDerivedStateFromError가 있는 클래스만 지원함)
class ErrorBoundary extends Component {
  state = { hasError: false }

  static getDerivedStateFromError() {
    return { hasError: true }
  }

  componentDidCatch(error, info) {
    console.error('[ErrorBoundary]', error, info)
  }

  handleReset = () => {
    this.setState({ hasError: false })
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className={styles.wrap}>
          <svg className={styles.icon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 9v4" /><path d="M12 17h.01" />
            <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          </svg>
          <p className={styles.title}>화면을 불러오는 중 오류가 발생했습니다</p>
          <p className={styles.desc}>페이지를 새로고침하거나 아래 버튼으로 다시 시도해주세요.</p>
          <button className={styles.retryBtn} onClick={this.handleReset}>다시 시도</button>
        </div>
      )
    }
    return this.props.children
  }
}

export default ErrorBoundary
