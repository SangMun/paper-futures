# XRP Dual Hedge Experiment

상태: 후보 (`EXP-015`). 이 디렉터리는 XRP_USDT 양방향·격리증거금 페이퍼
실험의 설계와 검증 자료를 모으는 작업 공간이다. 현재 활성 전략이나 실거래 주문
기능을 포함하지 않는다.

## 가설

같은 논리 시각에 동일한 총 증거금을 반씩 나눈 롱·숏을 진입한 뒤, 한 다리가
격리증거금 전액을 잃어 청산돼도 살아남은 다리의 추세 수익이 양쪽 체결 비용과
손실 다리의 손실을 넘어설 수 있는지 검증한다.

## v1 경계

- XRP_USDT, 동일 계약 수량, 양쪽 격리증거금, 페이퍼 체결만 다룬다.
- 양쪽 주문의 실제 평균 체결가 차이가 허용치를 넘거나 한쪽만 체결되면 즉시
  중립화한 비용을 손익에 포함한다.
- 일반 전략의 반대 포지션 차단 규칙은 유지한다. 같은 `pair_id`의 실험용 두
  다리에만 제한적으로 예외를 둔다.
- Gate.io 개인 API, API 키, 실주문·출금·실거래 전환 코드는 추가하지 않는다.
- 결과를 실거래 체결 성과나 투자 권유로 표현하지 않는다.

## 현재 페이퍼 포워드 구성

아래 네 가지 조합을 **서로 다른 1,000 USDT 가상 계좌**에서 동시에 관측한다. 어느
쪽도 방향을 미리 고르지 않으며, 진입하면 롱·숏 pair를 함께 연다.

| 계좌 태그 | 관측 간격 | pair를 여는 조건 |
|---|---:|---|
| `50x-continuous` (①) | 5초 | 실행 첫 유효 시세와, 이전 pair가 끝난 뒤의 다음 관측 시세 |
| `50x-squeeze_breakout` (③) | 5초 | 직전 확정 1분봉의 밴드폭이 그 전 100개 확정봉 하위 20%이고, 마지막 확정봉 종가가 직전 20봉 범위를 이탈 |
| `100x-continuous` (①) | 1초 | 실행 첫 유효 시세와, 이전 pair가 끝난 뒤의 다음 관측 시세 |
| `100x-squeeze_breakout` (③) | 1초 | 직전 확정 1분봉의 밴드폭이 그 전 100개 확정봉 하위 20%이고, 마지막 확정봉 종가가 직전 20봉 범위를 이탈 |

청산 감시는 1분봉 마감을 기다리지 않는다. 공개 1분봉 거래가는 100배는 기본 1초,
50배는 기본 5초마다 받아 트레일·이익보장 판정에 쓰고, 청산은 같은 관측에서 받은
Gate의 **Mark Price**, 공개 risk-limit 표의 **1단계 유지증거금률**, 계약의
청산 테이커 수수료로만 판정한다. Mark Price는
직전 관측값과 현재 관측값 사이의 확인 가능한 구간만 사용한다. 한쪽
청산 뒤 생존 다리가 `profit_lock_price`에 닿으면 그 즉시 비용 손익분기용 하드 스탑을
활성화하고, 그 위(롱)·아래(숏)에서만 ATR 트레일을 좁힌다.

```bash
./.venv/bin/python -m tools.paper_xrp_dual_hedge_live --minutes 60 \
  --poll-50-seconds 5 --poll-100-seconds 1 \
  --log-dir data/xrp_dual_hedge_mark_live
```

- 이 도구는 Gate 공개 캔들·계약 정보만 읽으며 개인 API·주문 API를 호출하지 않는다.
- Mark Price 판정 버전의 이벤트와 복구 상태는
  `data/xrp_dual_hedge_mark_live/<leverage>/<strategy>/`에 남긴다. 기존
  `xrp_dual_hedge_live` 결과와 섞지 않는다.
- 청산된 다리는 Gate 격리 규칙대로 배정 증거금까지만 잃는다.
- 첫 risk-limit 표의 한도를 넘는 pair는 시작하지 않고 관측만 계속한다. 상위 구간의 유지증거금
  deduction, 수동 증거금 조정, 펀딩·부분 청산은 개인 포지션 상태 없이는 재현할 수 없다.
- REST 폴링이므로 거래소 틱/웹소켓 실행은 아니다. 관측 사이의 Mark Price 움직임,
  스프레드와 실제 체결 순서는 여전히 확인할 수 없다.
- Discord 웹훅은 소스에 넣지 않는다. 실행 환경에 4개 실험별 URL을 설정한 경우에만
  해당 알림을 보내며, URL이 없어도 페이퍼 루프와 JSONL 기록은 정상 동작한다.
- 네 실험은 별도 전략 웹훅의 원본 이벤트 외에도, 기존 봇의 `system` 채널에 각각
  시작·UTC 일일 생존 상태를, `balance` 채널에 각각 진입·청산으로 바뀐 잔고를
  **개별 메시지**로 보낸다. 네 계좌를 하나의 요약 메시지로 합치지 않는다.

## Lightsail 배포

기존 `paper-futures.service`와 `paper-futures@.service`는 일반 `engine.loop` 실행용이다.
이 실험은 별도 서비스 [paper-xrp-dual-hedge.service](../../deploy/paper-xrp-dual-hedge.service)로
실행하며, 기존 봇과 함께 켜도 주문·상태 파일을 공유하지 않는다.

서버의 `/home/ubuntu/paper-futures/v2/.env.xrp-dual-hedge`에는 아래 네 환경변수를 둔다.
실제 URL은 로컬 저장소나 서비스 유닛에 넣지 않는다.

```text
DISCORD_XRP_DUAL_HEDGE_50_CONTINUOUS_WEBHOOK_URL=
DISCORD_XRP_DUAL_HEDGE_50_SQUEEZE_WEBHOOK_URL=
DISCORD_XRP_DUAL_HEDGE_100_CONTINUOUS_WEBHOOK_URL=
DISCORD_XRP_DUAL_HEDGE_100_SQUEEZE_WEBHOOK_URL=
```

서버에서 파일을 먼저 `ubuntu` 소유·0600 권한으로 만들고 값을 입력한다. 아래 명령은
이미 입력한 값을 지우지 않는다.

```bash
sudo touch /home/ubuntu/paper-futures/v2/.env.xrp-dual-hedge
sudo chown ubuntu:ubuntu /home/ubuntu/paper-futures/v2/.env.xrp-dual-hedge
sudo chmod 600 /home/ubuntu/paper-futures/v2/.env.xrp-dual-hedge
sudo -u ubuntu nano /home/ubuntu/paper-futures/v2/.env.xrp-dual-hedge
```

코드를 서버에 배포하고 가상환경 의존성을 갱신한 뒤 아래 순서로 등록한다.

```bash
sudo install -m 0644 deploy/paper-xrp-dual-hedge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now paper-xrp-dual-hedge.service
sudo systemctl status paper-xrp-dual-hedge.service
sudo journalctl -u paper-xrp-dual-hedge.service -f
```

Mark Price 버전의 상태·이벤트는 `data/xrp_dual_hedge_mark_live/`에 남고, 기존 runtime
S3 백업은 구버전·Mark Price 버전 디렉터리를 각각 일자별로 보관한다.

## 시작 순서

1. `MEASUREMENTS.md`의 EXP-015 정의와 비용 가정을 확정한다.
2. 합성 체결·청산 시나리오 테스트를 먼저 작성한다.
3. 체결 모형, 상태 복구, 엔진 경로, 리플레이 집계를 차례로 구현한다.
4. 기본 활성 전략에는 넣지 않은 채 리플레이와 페이퍼 포워드 기록을 분리한다.
5. 실행 결과와 결정은 `EXPERIMENTS.md`의 EXP-015에 기록한다.

구현 시 참조할 기존 파일은 [SOURCE_REFERENCES.md](SOURCE_REFERENCES.md)에 정리한다.
