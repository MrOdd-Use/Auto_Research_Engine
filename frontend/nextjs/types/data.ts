export interface BaseData {
  type: string;
}

export interface BasicData extends BaseData {
  type: 'basic';
  content: string;
}

export interface LanggraphButtonData extends BaseData {
  type: 'langgraphButton';
  link: string;
}

export interface DifferencesData extends BaseData {
  type: 'differences';
  content: string;
  output: string;
}

export interface QuestionData extends BaseData {
  type: 'question';
  content: string;
}

export interface ChatData extends BaseData {
  type: 'chat';
  content: string;
  metadata?: any; // For storing search results and other contextual information
}

export type Data = BasicData | LanggraphButtonData | DifferencesData | QuestionData | ChatData;

export interface MCPConfig {
  name: string;
  command: string;
  args: string[];
  env: Record<string, string>;
}

export interface ChatBoxSettings {
  report_type: string;
  report_source: string;
  tone: string;
  domains: string[];
  defaultReportType: string;
  layoutType: string;
  mcp_enabled: boolean;
  mcp_configs: MCPConfig[];
  mcp_strategy?: string;
}

export interface Domain {
  value: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  timestamp?: number;
  metadata?: any; // For storing search results and other contextual information
}

export interface ResearchHistoryItem {
  id: string;
  question: string;
  answer: string;
  timestamp: number;
  orderedData: Data[];
  chatMessages?: ChatMessage[];
  status?: 'running' | 'completed' | 'failed';
  workflow_available?: boolean;
  current_session_id?: string;
  last_successful_session_id?: string;
} 

export interface WorkflowCheckpointNode {
  checkpoint_id: string;
  node_name: string;
  display_name?: string;
  status: string;
  step_order: number;
  rerunnable: boolean;
  summary?: any;
  scope: string;
  section_key?: string;
}

export interface WorkflowSectionTree {
  section_key: string;
  section_index: number;
  section_title: string;
  scope: string;
  checkpoints: WorkflowCheckpointNode[];
}

export interface WorkflowCheckpointTree {
  global_nodes: WorkflowCheckpointNode[];
  sections: WorkflowSectionTree[];
}

export interface WorkflowSessionSummary {
  session_id: string;
  parent_session_id?: string | null;
  root_session_id?: string | null;
  rerun_from_checkpoint_id?: string | null;
  round_index: number;
  status: 'running' | 'completed' | 'failed';
  created_at: string;
  updated_at?: string;
  note?: string | null;
  target?: any;
}

export interface WorkflowSelectedSession extends WorkflowSessionSummary {
  answer: string;
  ordered_data: Data[];
  checkpoints_tree: WorkflowCheckpointTree;
}

export interface WorkflowResponse {
  report_id: string;
  workflow_available: boolean;
  legacy_reason?: string | null;
  current_session_id?: string | null;
  last_successful_session_id?: string | null;
  sessions: WorkflowSessionSummary[];
  selected_session?: WorkflowSelectedSession | null;
}
