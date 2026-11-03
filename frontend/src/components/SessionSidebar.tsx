import { motion } from "motion/react";
import { NavLink } from "react-router-dom";
import { Session } from "../types/api";

interface SessionSidebarProps {
  sessions: Session[];
  selected?: string;
  collapsed: boolean;
  onToggle: () => void;
  onSelect: (id: string) => void;
  onCreate: () => void;
  onArchive: (id: string) => void;
  onRename: (session: Session) => void;
}

/** 会话侧栏：可收缩成 64px 图标条（宽度由 motion 动画驱动，内容随宽度淡出） */
export function SessionSidebar({ sessions, selected, collapsed, onToggle, onSelect, onCreate, onArchive, onRename }: SessionSidebarProps) {
  return <motion.aside
    className={`sidebar${collapsed ? " collapsed" : ""}`}
    initial={false}
    animate={{ width: collapsed ? 64 : 264 }}
    transition={{ type: "spring", stiffness: 300, damping: 32 }}
  >
    <button className="brand" onClick={onToggle} title={collapsed ? "展开侧栏" : "收起侧栏"}>
      <span className="brand-mark">D</span>
      <div className="brand-text collapse-fade"><strong>DeepTrace</strong><small>研究工作台</small></div>
    </button>
    <button className="new-session" onClick={onCreate} title={collapsed ? "新建会话" : undefined}>{collapsed ? "＋" : "＋ 新建会话"}</button>
    <div className="sidebar-label collapse-fade">研究会话</div>
    <div className="session-list">
      {sessions.map((session) => (
        <div key={session.session_id} className={`session-row ${selected === session.session_id ? "selected" : ""}`}>
          <button onClick={() => onSelect(session.session_id)} title={collapsed ? session.title : undefined}>
            <span className="session-dot">{session.title.slice(0, 1)}</span>
            <span className="collapse-fade">{session.title}</span>
            <small className="collapse-fade">{session.status === "archived" ? "已归档" : new Date(session.updated_at).toLocaleDateString()}</small>
          </button>
          {!collapsed && session.status === "active" && <><button className="archive" onClick={() => onRename(session)} title="重命名">✎</button><button className="archive" onClick={() => onArchive(session.session_id)} title="归档">⌁</button></>}
        </div>
      ))}
    </div>
    <nav>
      <NavLink to="/" title="研究工作台"><span className="nav-icon">⌁</span>{!collapsed && <span className="nav-label">研究工作台</span>}</NavLink>
      <NavLink to="/memories" title="长期记忆"><span className="nav-icon">◇</span>{!collapsed && <span className="nav-label">长期记忆</span>}</NavLink>
      <NavLink to="/skills" title="研究技能"><span className="nav-icon">△</span>{!collapsed && <span className="nav-label">研究技能</span>}</NavLink>
      <NavLink to="/evolution" title="复盘与评测"><span className="nav-icon">◎</span>{!collapsed && <span className="nav-label">复盘与评测</span>}</NavLink>
      <NavLink to="/status" title="系统状态"><span className="nav-icon">○</span>{!collapsed && <span className="nav-label">系统状态</span>}</NavLink>
    </nav>
  </motion.aside>;
}
