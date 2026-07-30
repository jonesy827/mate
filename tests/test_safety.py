from matebridge.safety import is_destructive

DESTRUCTIVE_SAMPLES = [
    "git push --force origin main",
    "about to force-push the branch",
    "TRUNCATE TABLE sessions;",
    "truncate table users",
    "DROP TABLE customers",
    "drop   table customers",
    "terraform apply -auto-approve",
    "terraform destroy",
    "deploy to prod now?",
    "this will touch the production database",
    "rm -rf /var/lib/data",
    "rm -r ./build",  # -rf? also flags plain rm -r; deliberate, safety-biased
    "git reset --hard HEAD~3",
    "run with --force flag",
]

SAFE_SAMPLES = [
    "echo hello world",
    "npm install && npm test",
    "git commit -m 'add product catalog page'",  # 'product' is not 'prod'
    "trying to reproduce the bug",
    "cargo build --release",
    "warm -rf is not a command",  # 'rm' must be word-boundary anchored
    "git push origin feature/reset-form",
    "pytest tests/ -k 'not slow'",
    "informal chat about the workforce",
]


def test_destructive_samples_flagged():
    for s in DESTRUCTIVE_SAMPLES:
        assert is_destructive(s), f"should be flagged: {s!r}"


def test_safe_samples_pass():
    for s in SAFE_SAMPLES:
        assert not is_destructive(s), f"false positive: {s!r}"
