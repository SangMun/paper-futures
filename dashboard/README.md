# Paper Desk 대시보드

페이퍼 봇의 체결 기록과 Gate.io 공개 캔들을 읽기 전용으로 보여주는 대시보드.
봇 프로세스와 분리돼 있어서, 대시보드가 죽어도 매매 루프에는 영향이 없다.

| 경로 | 내용 |
|---|---|
| `/` | BTC 회귀채널 복귀 전략 |
| `/btc-25x` | BTC 국면 추종 25배 전략 |
| `/xrp-patterns` | XRP 차트 패턴 전략 |
| `/api/btc-live`, `/api/xrp-live` | 최근 시세 (Gate 공개 API) |

## 데이터 흐름

```
서버 data/<계좌>/  ─(rsync 또는 S3 백업)→  로컬 data/
                                            └ scripts/export_dashboard_data.py
                                               → public/btc-dashboard.json (스냅샷)
```

```bash
# 서버 파일에서 갱신
BOT_HOST=ubuntu@<서버> SSH_KEY=<키 경로> dashboard/scripts/refresh_from_server.sh
# 또는 S3 일일 백업에서 갱신
BACKUP_BUCKET=<버킷> dashboard/scripts/refresh_from_s3.sh
```

## 개발

Node.js `>=22.13.0`. [vinext](https://github.com/cloudflare/vinext) 기반이다.

```bash
npm install
npm run dev
npm run lint
npm test        # build 후 node --test
```

`db/`와 `drizzle.config.ts`는 호스팅 템플릿의 D1 바인딩 자리다. 지금은 DB를 쓰지 않는다.
