# 1단계: 빌드 환경 구성 (Gradle 래퍼와 소스코드를 복사하여 빌드)
FROM eclipse-temurin:17-jdk-alpine AS builder
WORKDIR /app

# Gradle 래퍼와 설정 파일들을 먼저 복사하여 종속성을 캐싱
COPY gradlew .
COPY gradle gradle
COPY build.gradle .
COPY settings.gradle .

# 스크립트 실행 권한 부여 및 의존성 다운로드
RUN chmod +x ./gradlew
RUN ./gradlew dependencies --no-daemon

# 소스 코드 복사 후 빌드
COPY src src
RUN ./gradlew bootJar -x test --no-daemon

# 2단계: 실행 환경 구성 (빌드된 jar 파일만 가져와서 실행)
FROM eclipse-temurin:17-jre-alpine
WORKDIR /app

# 빌드 단계에서 생성된 jar 파일을 실행 환경으로 복사
COPY --from=builder /app/build/libs/*.jar app.jar

# 운영 환경 프로필(prod)을 기본으로 하여 앱 실행
ENTRYPOINT ["java", "-jar", "app.jar", "--spring.profiles.active=prod"]