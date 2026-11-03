import { useEffect, useRef, useState } from "react";
import { API_BASE, api } from "../api/client";
import { Run, RunEvent } from "../types/api";
import { runIsSettled } from "./runStatus";

/** 统一的后端状态轮询间隔（毫秒） */
export const RUN_POLL_MS = 2000;

export function useRunEvents(runId?: string | null) {
  const [generation, setGeneration] = useState(0);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [connection, setConnection] = useState<"idle" | "connected" | "reconnecting">("idle");
  const cursor = useRef(0);
  const runRef = useRef<Run | null>(null); // 轮询闭包读最新状态，避免过期闭包

  useEffect(() => {
    if (!runId) { setEvents([]); setRun(null); setConnection("idle"); runRef.current = null; return; }
    let closed = false; let source: EventSource | undefined;
    let polling = false;
    let reconnectTimer: number | undefined; let pollTimer: number | undefined;
    const sync = () => api.run(runId).then((value) => {
      if (closed) return undefined;
      runRef.current = value; setRun(value); return value;
    }).catch(() => undefined);
    const poll = () => {
      if (polling || closed) return;
      if (runIsSettled(runRef.current)) { window.clearInterval(pollTimer); return; }
      polling = true;
      sync().finally(() => { polling = false; });
    };
    const connect = () => {
      if (closed) return;
      source = new EventSource(`${API_BASE}/runs/${runId}/events?after_event_id=${cursor.current}`);
      source.onopen = () => { if (!closed) setConnection("connected"); };
      const handle = (message: MessageEvent) => {
        if (closed) return;
        const data = JSON.parse(message.data) as RunEvent;
        if (typeof data.event_id === "number") cursor.current = Math.max(cursor.current, data.event_id);
        setEvents((items) => items.some((item) => item.event_id === data.event_id) ? items : [...items, data]);
        if (String(data.event_type).startsWith("run.")) sync();
      };
      ["run.queued", "run.started", "run.stage_changed", "run.pause_requested", "run.paused", "run.resumed", "run.interrupted", "run.resume_requested", "run.recovery_waiting", "run.recovery_deferred", "run.completed", "run.failed", "run.cancelled", "run.budget_exhausted", "run.needs_user_input", "plan.created", "plan.revised", "task.started", "task.completed", "task.failed", "tool.started", "tool.completed", "tool.failed", "evidence.added", "report.completed", "verification.completed", "agent.progress", "iteration.completed", "delivery.requirements.created", "delivery.review.completed", "delivery.supplement.completed", "delivery.supplement.failed"].forEach((name) => source?.addEventListener(name, handle));
      source.onerror = () => {
        if (closed) return;
        source?.close(); setConnection("reconnecting");
        sync().then((value) => {
          if (closed) return;
          if (runIsSettled(value)) setConnection("idle");
          else if (!closed) reconnectTimer = window.setTimeout(connect, 1_000);
        });
      };
    };
    setEvents([]); cursor.current = 0; runRef.current = null; sync(); connect();
    pollTimer = window.setInterval(poll, RUN_POLL_MS);
    return () => { closed = true; source?.close(); if (reconnectTimer) window.clearTimeout(reconnectTimer); if (pollTimer) window.clearInterval(pollTimer); };
  }, [runId, generation]);

  return {
    events, run, connection,
    refresh: () => runId ? api.run(runId).then((value) => { runRef.current = value; setRun(value); }) : Promise.resolve(),
    restart: () => setGeneration((value) => value + 1),
  };
}
