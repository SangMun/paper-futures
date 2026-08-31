"""테스트 전역 안전장치.

**테스트가 실제 상태 파일과 로그를 건드리지 못하게 막는다.**

Engine._persist()는 config.settings.STATE_PATH에 쓴다. 테스트에서 bot_state를
주입해도 저장 경로까지 바뀌지는 않아서, run_tick 테스트가 data/state.json에
"test" 전략의 연속 손절 카운터를 남겼다. 페이퍼 계좌 상태를 테스트가
덮어쓰면 잔고와 열린 포지션을 잃는다.

개별 테스트가 기억해서 막을 일이 아니라 여기서 구조적으로 막는다.
"""

from __future__ import annotations

import pytest

from engine import state as st


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """상태 저장을 임시 경로로 돌린다."""

    target = tmp_path / "state.json"
    real_save = st.save

    def guarded(bot_state, path=None):
        return real_save(bot_state, path or target)

    monkeypatch.setattr(st, "save", guarded)
    yield
