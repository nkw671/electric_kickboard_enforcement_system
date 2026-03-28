import { Link, useLocation } from 'react-router-dom'
import styles from './Layout.module.css'

function Layout({ children }) {
  const location = useLocation()

  return (
    <div className={styles.container}>
      <header className={styles.header}>
        <div className={styles.logo}>전동킥보드 단속 시스템</div>
        <nav className={styles.nav}>
          <Link
            to="/"
            className={location.pathname === '/' ? styles.active : ''}
          >
            메인
          </Link>
          <Link
            to="/violations"
            className={location.pathname === '/violations' ? styles.active : ''}
          >
            위반 기록
          </Link>
        </nav>
      </header>
      <main className={styles.main}>{children}</main>
    </div>
  )
}

export default Layout
