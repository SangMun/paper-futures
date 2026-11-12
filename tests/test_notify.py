

def test_code_revision_falls_back_without_git(monkeypatch):
    import subprocess

    from reporting import notify

    def boom(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert notify.code_revision() == "unknown"


def test_code_revision_marks_dirty(monkeypatch):
    import subprocess

    from reporting import notify

    outputs = iter(["abc1234\n", " M engine/loop.py\n"])

    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=next(outputs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert notify.code_revision() == "abc1234+dirty"
