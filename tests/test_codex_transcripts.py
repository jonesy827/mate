import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mate import transcripts as tr
from mate.agent import Mate
from mate.watcher import FleetWatcher

SID = '12345678-1234-1234-1234-123456789abc'


def rollout(path, sid=SID):
    records = [{'type': 'session_meta', 'payload': {'id': sid}}]
    def msg(text, phase, mid):
        return {'type': 'response_item', 'payload': {
            'type': 'message', 'role': 'assistant', 'phase': phase, 'id': mid,
            'content': [{'type': 'output_text', 'text': text}]}}
    records += [msg('progress', 'commentary', 'a'), msg('first answer', 'final_answer', 'b'),
                msg('first answer', 'final_answer', 'b'),
                {'type': 'response_item', 'payload': {'type': 'reasoning', 'text': 'secret'}},
                {'type': 'event_msg', 'payload': {'type': 'agent_message', 'message': 'duplicate'}},
                msg('second answer', 'final', 'c')]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(json.dumps(r) for r in records) + '\n{"partial":')
    return path


def test_reads_final_answers_once_and_ignores_partial_write(tmp_path):
    p = rollout(tmp_path / 'session.jsonl')
    assert tr.read_codex_replies(p, SID, 2) == ['first answer', 'second answer']
    assert tr.read_codex_replies(p, SID, 1) == ['second answer']
    with pytest.raises(OSError, match='identity'):
        tr.read_codex_replies(p, 'different', 1)


def test_index_points_to_resumed_segment_and_does_not_guess(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, 'CODEX_STATE_DIR', tmp_path)
    p = rollout(tmp_path / 'sessions' / 'resumed.jsonl')
    with sqlite3.connect(tmp_path / 'state_5.sqlite') as conn:
        conn.execute('create table threads (id text, rollout_path text)')
        conn.execute('insert into threads values (?, ?)', (SID, str(p)))
    assert tr.codex_transcript_path(SID) == p
    with pytest.raises(OSError):
        tr.codex_transcript_path('unknown')
    with pytest.raises(OSError):
        tr.codex_transcript_path('../secrets')


def test_unique_file_without_index(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, 'CODEX_STATE_DIR', tmp_path)
    p = rollout(tmp_path / 'sessions' / f'rollout-date-{SID}.jsonl')
    assert tr.codex_transcript_path(SID) == p
    rollout(tmp_path / 'sessions' / 'other' / p.name)
    with pytest.raises(OSError, match='unique'):
        tr.codex_transcript_path(SID)


async def test_report_recovers_session_from_pane_details(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, 'CODEX_STATE_DIR', tmp_path)
    rollout(tmp_path / 'sessions' / f'rollout-date-{SID}.jsonl')
    agent = {'pane_id': 'p', 'agent': 'codex'}
    herdr = SimpleNamespace(snapshot=AsyncMock(return_value={'agents': [agent]}),
        call=AsyncMock(return_value={'pane': dict(agent, agent_session={'kind': 'id', 'value': SID})}),
        read_pane=AsyncMock())
    assert await Mate(herdr)._agent_replies('p') == 'second answer'
    herdr.read_pane.assert_not_awaited()


async def test_missing_identity_automatically_reads_same_pane_and_announces():
    agent = {'pane_id': 'p', 'agent': 'codex'}
    herdr = SimpleNamespace(snapshot=AsyncMock(return_value={'agents': [agent]}),
        call=AsyncMock(return_value={'pane': agent}),
        read_pane=AsyncMock(return_value='Implemented networking. Tests passed.'))
    mate = Mate(herdr)
    mate.delegated.add('p')
    mate.delegated.mark_started('p')
    say = AsyncMock()
    await FleetWatcher(herdr, mate, say).announce('idle', 'p', {})
    herdr.read_pane.assert_awaited_once_with('p', 100)
    spoken = say.call_args.args[0]
    assert 'screen' in spoken and 'Implemented networking' in spoken
    assert 'Want me to read' not in spoken
