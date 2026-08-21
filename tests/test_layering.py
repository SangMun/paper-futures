"""계층 경계 강제.

**감지 장치다.** 미마감 데이터가 지표·전략 쪽으로 새면 조용히 망가진다.
같은 시점을 다시 계산할 때 값이 달라지는데, 그건 실행 중에는 드러나지 않고
과거 검증 결과와 실시간 성과가 갈릴 때에야 알게 된다.

수입 경로를 막아 두면 실수가 커밋 시점에 잡힌다.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent

# 진행 중인 봉을 다루는 모듈. 실행 경로 전용이다.
LIVE_MODULES = {"core.live"}

# 이 디렉터리들은 확정봉만 본다.
BAR_ONLY_DIRS = ("indicators", "strategies")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)

    return found


def python_files(*dirs: str) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        out.extend(p for p in (ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)
    return out


@pytest.mark.parametrize(
    "path", python_files(*BAR_ONLY_DIRS), ids=lambda p: str(p.relative_to(ROOT))
)
def test_지표와_전략은_실시간_시세를_import하지_않는다(path: Path):
    leaked = imported_modules(path) & LIVE_MODULES

    assert not leaked, (
        f"{path.relative_to(ROOT)}가 {leaked}를 import한다. "
        "지표·전략은 확정봉만 본다. 진행 중인 봉이 필요하면 "
        "engine/ 쪽에서 처리할 것."
    )


def test_감시_대상_모듈이_실제로_존재한다():
    """모듈 이름이 바뀌면 위 테스트가 조용히 무력화된다."""

    for module in LIVE_MODULES:
        rel = Path(module.replace(".", "/") + ".py")
        assert (ROOT / rel).exists(), f"{module}이 없다. LIVE_MODULES를 갱신할 것"


def test_검사_대상_파일이_비어있지_않다():
    """rglob가 아무것도 못 잡으면 테스트가 항상 통과한다."""

    files = python_files(*BAR_ONLY_DIRS)

    assert len(files) > 5, f"검사 대상이 {len(files)}개뿐이다. 경로 확인 필요"
