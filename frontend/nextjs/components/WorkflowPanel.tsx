import React, { useMemo, useState } from 'react';
import {
  WorkflowCheckpointNode,
  WorkflowResponse,
  WorkflowSectionTree,
} from '@/types/data';

interface WorkflowPanelProps {
  workflow: WorkflowResponse | null;
  loading?: boolean;
  onSelectSession?: (sessionId: string) => void;
  onRerunCheckpoint?: (checkpointId: string, note: string) => void;
}

interface PendingRerun {
  checkpointId: string;
  title: string;
}

const SessionButton = ({
  label,
  active,
  onClick,
  status,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  status: string;
}) => (
  <button
    onClick={onClick}
    className={`rounded-md border px-3 py-1.5 text-sm transition-colors ${
      active
        ? 'border-teal-400 bg-teal-500/20 text-teal-100'
        : 'border-gray-700 bg-gray-900/60 text-gray-300 hover:border-gray-500'
    }`}
  >
    {label} | {status}
  </button>
);

const CheckpointRow = ({
  checkpoint,
  prefix,
  disabled,
  onRerun,
}: {
  checkpoint: WorkflowCheckpointNode;
  prefix?: string;
  disabled?: boolean;
  onRerun?: (checkpoint: WorkflowCheckpointNode) => void;
}) => {
  const summaryText = useMemo(() => {
    if (!checkpoint.summary) return '';
    const entries = Object.entries(checkpoint.summary);
    return entries
      .map(([key, value]) => `${key}: ${Array.isArray(value) ? value.join(', ') : String(value)}`)
      .join(' | ');
  }, [checkpoint.summary]);

  return (
    <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3">
      <div className="flex items-center justify-between gap-4">
        <div>
          <div className="text-sm font-medium text-white">
            {prefix ? `${prefix} | ` : ''}
            {checkpoint.display_name || checkpoint.node_name}
          </div>
          <div className="mt-1 text-xs text-gray-400">
            {checkpoint.status}
            {summaryText ? ` | ${summaryText}` : ''}
          </div>
        </div>
        {checkpoint.rerunnable && onRerun && (
          <button
            disabled={disabled}
            onClick={() => onRerun(checkpoint)}
            className="rounded-md border border-teal-700 px-3 py-1.5 text-xs text-teal-200 transition-colors hover:bg-teal-800/30 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Rerun from Checkpoint
          </button>
        )}
      </div>
    </div>
  );
};

const SectionBlock = ({
  section,
  disabled,
  onRerun,
}: {
  section: WorkflowSectionTree;
  disabled?: boolean;
  onRerun?: (checkpoint: WorkflowCheckpointNode) => void;
}) => (
  <div className="rounded-lg border border-gray-800/80 bg-black/20 p-4">
    <div className="mb-3 text-sm font-semibold text-sky-100">{section.section_title}</div>
    <div className="space-y-2">
      {section.checkpoints.map((checkpoint) => (
        <CheckpointRow
          key={checkpoint.checkpoint_id}
          checkpoint={checkpoint}
          prefix="Section"
          disabled={disabled}
          onRerun={onRerun}
        />
      ))}
    </div>
  </div>
);

export default function WorkflowPanel({
  workflow,
  loading = false,
  onSelectSession,
  onRerunCheckpoint,
}: WorkflowPanelProps) {
  const [pendingRerun, setPendingRerun] = useState<PendingRerun | null>(null);
  const [note, setNote] = useState('');

  if (!workflow) {
    return null;
  }

  const selectedSession = workflow.selected_session;

  return (
    <>
      <div className="container mt-5 rounded-lg border border-gray-700/40 bg-black/30 p-5 shadow-lg backdrop-blur-md">
        <div className="flex items-center justify-between gap-4 pb-4">
          <div>
            <h3 className="text-base font-bold uppercase leading-[152.5%] text-white">Rerun from Checkpoint</h3>
            <p className="mt-1 text-sm text-gray-400">
              {workflow.workflow_available
                ? 'Browse prior rounds and start Rerun from Checkpoint from any node.'
                : workflow.legacy_reason || 'This report does not support Rerun from Checkpoint yet.'}
            </p>
          </div>
        </div>

        {workflow.sessions.length > 0 && (
          <div className="mb-4 flex flex-wrap gap-2">
            {workflow.sessions.map((session) => (
              <SessionButton
                key={session.session_id}
                label={`Session ${session.round_index}`}
                status={session.status}
                active={session.session_id === selectedSession?.session_id}
                onClick={() => onSelectSession?.(session.session_id)}
              />
            ))}
          </div>
        )}

        {selectedSession && (
          <div className="space-y-4">
            <div className="rounded-md border border-gray-800 bg-gray-950/60 p-3 text-sm text-gray-300">
              Active round: Session {selectedSession.round_index}
              {selectedSession.note ? ` | note: ${selectedSession.note}` : ''}
            </div>

            <div className="space-y-2">
              {(selectedSession.checkpoints_tree?.global_nodes || []).map((checkpoint) => (
                <CheckpointRow
                  key={checkpoint.checkpoint_id}
                  checkpoint={checkpoint}
                  disabled={loading}
                  onRerun={(item) =>
                    setPendingRerun({
                      checkpointId: item.checkpoint_id,
                      title: item.display_name || item.node_name,
                    })
                  }
                />
              ))}
            </div>

            {(selectedSession.checkpoints_tree?.sections || []).length > 0 && (
              <div className="space-y-3">
                <div className="text-sm font-semibold text-white">Section Checkpoints</div>
                {(selectedSession.checkpoints_tree?.sections || []).map((section) => (
                  <SectionBlock
                    key={section.section_key}
                    section={section}
                    disabled={loading}
                    onRerun={(item) =>
                      setPendingRerun({
                        checkpointId: item.checkpoint_id,
                        title: `${section.section_title} / ${item.display_name || item.node_name}`,
                      })
                    }
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {pendingRerun && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4">
          <div className="w-full max-w-xl rounded-xl border border-gray-700 bg-gray-950 p-6 shadow-2xl">
            <div className="text-lg font-semibold text-white">Rerun from Checkpoint</div>
            <div className="mt-2 text-sm text-gray-400">{pendingRerun.title}</div>
            <textarea
              value={note}
              onChange={(event) => setNote(event.target.value)}
              placeholder="Add optional instructions for this Rerun from Checkpoint."
              className="mt-4 min-h-[140px] w-full rounded-lg border border-gray-700 bg-black/40 p-3 text-sm text-white outline-none focus:border-teal-500"
            />
            <div className="mt-4 flex justify-end gap-3">
              <button
                onClick={() => {
                  setPendingRerun(null);
                  setNote('');
                }}
                className="rounded-md border border-gray-700 px-4 py-2 text-sm text-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={() => {
                  onRerunCheckpoint?.(pendingRerun.checkpointId, note);
                  setPendingRerun(null);
                  setNote('');
                }}
                className="rounded-md border border-teal-600 bg-teal-700/20 px-4 py-2 text-sm text-teal-100"
              >
                Start Rerun from Checkpoint
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
