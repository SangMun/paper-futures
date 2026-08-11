"""환경변수 로딩.

비밀값(웹훅 URL, 나중의 API 키)은 코드에 넣지 않는다.
`.env`는 .gitignore에 있고, 실제 환경변수가 우선한다.

python-dotenv를 쓰지 않는 이유는 의존성 하나를 위해 붙이기엔 기능이
`KEY=VALUE` 파싱뿐이기 때문이다.
"""

from __future__ import annotations

import os
from pathlib import Path


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

_loaded = False


def load(path: Path | None = None, override: bool = False) -> dict[str, str]:
    """`.env`를 읽어 os.environ에 채운다. 파일이 없으면 조용히 넘어간다.

    override=False면 이미 설정된 환경변수를 덮어쓰지 않는다.
    운영 환경에서 진짜 환경변수로 주입한 값이 파일에 밀리면 안 된다.
    """

    global _loaded
    path = path or ENV_PATH
    values: dict[str, str] = {}

    if not path.exists():
        _loaded = True
        return values

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value

    _loaded = True
    return values


def get(name: str, default: str | None = None) -> str | None:
    """환경변수 조회. 첫 호출에서 .env를 한 번 읽는다."""

    if not _loaded:
        load()
    return os.environ.get(name, default)
