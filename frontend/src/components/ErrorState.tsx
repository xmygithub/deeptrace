export function ErrorState({ error, onRetry }: {error: unknown; onRetry?: () => void}) {
  if (!error) return null;
  const value = error as {code?: string; message?: string; retryable?: boolean};
  const advice: Record<string, string> = {TAVILY_API_KEY_MISSING: "请在后端 .env 配置 TAVILY_API_KEY。", SOURCE_UNAVAILABLE: "请检查所选信息源配置。", NO_SOURCE_EVIDENCE: "请导入与问题相关的私域资料，或切换到联网搜索后重新提交。", INSUFFICIENT_SOURCE_EVIDENCE: "当前私域证据不足以支撑所需报告深度；请补充相关资料，或切换到联网搜索后重新提交。", NETWORK_ERROR: "请确认 FastAPI 后端正在运行。"};
  return <div className="error-state"><strong>{value.code ?? "请求失败"}</strong><span>{value.message ?? String(error)}</span>{advice[value.code ?? ""] && <small>{advice[value.code ?? ""]}</small>}{onRetry && <button onClick={onRetry}>重试</button>}</div>;
}
