import { Run, RunEvent } from "../types/api";

export function runIsSettled(run?: Pick<Run, "status" | "resume_requested"> | null): boolean {
  if (!run) return false;
  return ["completed", "failed", "cancelled", "budget_exhausted", "paused"].includes(run.status)
    && !(run.status === "paused" && run.resume_requested);
}

export function recoveryNotice(events: RunEvent[]): {waiting: boolean; message: string} | null {
  const latest = [...events].reverse().find(event => [
    "run.recovery_waiting", "run.recovery_deferred", "run.resumed", "run.started",
    "run.completed", "run.failed", "run.cancelled", "run.paused", "run.budget_exhausted",
  ].includes(event.event_type));
  if (latest?.event_type === "run.recovery_waiting") return {
    waiting: true, message: "正在等待原执行器的运行租约到期，系统会自动重试接管。已保存的进度会保留。",
  };
  if (latest?.event_type === "run.recovery_deferred") return {
    waiting: false, message: "等待接管超时，系统未强制抢占运行租约。请确认其他执行器已停止后重试恢复。",
  };
  return null;
}
