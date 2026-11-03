import { Capabilities, SourceMode } from "../types/api";

export function SourceSelector({ value, onChange, capabilities }: {value: SourceMode; onChange: (mode: SourceMode) => void; capabilities?: Capabilities}) {
  return <div className="source-selector" aria-label="信息源">
    {(["graphrag", "web"] as SourceMode[]).map((mode) => {
      const available = capabilities?.sources[mode]?.available ?? true;
      return <label key={mode} className={`${value === mode ? "selected" : ""} ${!available ? "disabled" : ""}`} title={capabilities?.sources[mode]?.reason ?? undefined}>
        <input type="radio" name="source" value={mode} checked={value === mode} disabled={!available} onChange={() => onChange(mode)} />
        <span>{mode === "graphrag" ? "私有数据库" : "联网搜索"}</span>
      </label>;
    })}
  </div>;
}
