import js from '@eslint/js'
import globals from 'globals'
import react from 'eslint-plugin-react'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat['recommended-latest'],
      reactRefresh.configs.vite,
    ],
    plugins: { react },
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      // 코어 no-unused-vars가 eslint-scope에 jsx 옵션을 전달받지 못해
      // JSX 태그로만 쓰인 컴포넌트 import(App.jsx의 <Routes>, StatsPage.jsx의
      // Recharts 컴포넌트 등)를 오탐 처리하는 문제를 보정.
      'react/jsx-uses-vars': 'error',
    },
  },
])
