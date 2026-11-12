# Existing Project References

이 실험은 아래 실행 원본을 **참조**한다. 코드 복사본을 만들지 않아야 수정이 한
경로에 남고, 실시간과 리플레이가 같은 엔진을 사용한다.

| 목적 | 실행 원본 |
|---|---|
| 공통 작업·안전 지침 | [`../../AGENTS.md`](../../AGENTS.md) |
| 측정 정의·비용 가정 | [`../../MEASUREMENTS.md`](../../MEASUREMENTS.md), `config/settings.py::FEES` |
| 실험 색인·결정 기록 | [`../../EXPERIMENTS.md`](../../EXPERIMENTS.md) |
| 시장가 체결·수수료·순손익 | `engine/broker.py` |
| 포지션 상태·재시작 복구 | `engine/state.py` |
| 진입·반대 포지션 차단·청산 루프 | `engine/loop.py` |
| 리스크 상한과 사이징 | `risk/limits.py`, `risk/sizing.py` |
| 리플레이 집계 | `engine/replay.py::ReplayResult` |
| 활성 종목·전략 | `config/symbols.py::ACTIVE` |

새 구현은 이 목록의 실행 원본을 수정하고, 이 폴더에는 해당 변경의 설계 근거와
검증 자료를 남긴다.
