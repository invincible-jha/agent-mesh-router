/**
 * TypeScript interfaces for the agent-mesh-router service.
 *
 * Mirrors the Python dataclasses and enums defined in:
 *   agent_mesh_router.messages.types
 *   agent_mesh_router.workflows.base
 *   agent_mesh_router.routing.table
 *   agent_mesh_router.fleet.registry
 *   agent_mesh_router.resilience.circuit_breaker
 *
 * All interfaces use readonly fields to match Python's frozen dataclasses.
 */

// ---------------------------------------------------------------------------
// Message types and priority
// ---------------------------------------------------------------------------

/**
 * Discriminator for inter-agent message semantics.
 * Maps to MessageType enum in agent_mesh_router.messages.types.
 */
export type MessageType =
  | "task"
  | "query"
  | "response"
  | "result"
  | "handoff"
  | "broadcast"
  | "heartbeat"
  | "error"
  | "cancel"
  | "ack";

/**
 * Message priority levels compatible with asyncio.PriorityQueue ordering.
 * Lower numeric values are processed first.
 * Maps to Priority enum in agent_mesh_router.messages.types.
 */
export type MessagePriority = "CRITICAL" | "HIGH" | "NORMAL" | "LOW" | "BATCH";

// ---------------------------------------------------------------------------
// Message
// ---------------------------------------------------------------------------

/**
 * An inter-agent message routed through the mesh.
 * Encompasses all MessageType variants.
 */
export interface Message {
  /** Unique message identifier (UUID4). */
  readonly message_id: string;
  /** Discriminator for message semantics. */
  readonly message_type: MessageType;
  /** Agent identifier of the sender. */
  readonly sender_id: string;
  /** Agent identifier of the intended recipient (empty for broadcasts). */
  readonly recipient_id: string;
  /** Message payload — arbitrary key-value pairs. */
  readonly payload: Readonly<Record<string, unknown>>;
  /** Priority level controlling queue ordering. */
  readonly priority: MessagePriority;
  /** ISO-8601 UTC timestamp when the message was created. */
  readonly created_at: string;
  /** Optional correlation ID linking related messages (e.g. request/response). */
  readonly correlation_id?: string;
  /** Optional topic string used for broadcast fan-out. */
  readonly topic?: string;
}

// ---------------------------------------------------------------------------
// SendMessageRequest
// ---------------------------------------------------------------------------

/**
 * Request payload for dispatching a message into the mesh.
 */
export interface SendMessageRequest {
  /** Target agent identifier. */
  readonly recipient_id: string;
  /** Message type discriminator. */
  readonly message_type: MessageType;
  /** Message payload. */
  readonly payload: Readonly<Record<string, unknown>>;
  /** Priority level (default "NORMAL"). */
  readonly priority?: MessagePriority;
  /** Optional correlation ID for request/response linking. */
  readonly correlation_id?: string;
}

// ---------------------------------------------------------------------------
// Workflow pattern and steps
// ---------------------------------------------------------------------------

/**
 * Workflow coordination pattern.
 * Maps to WorkflowExecutor subclass names in agent_mesh_router.workflows.
 */
export type WorkflowPattern =
  | "sequential"
  | "parallel"
  | "hierarchical"
  | "competitive"
  | "consensus";

/**
 * A single step within a workflow.
 * Maps to WorkflowStep dataclass in agent_mesh_router.workflows.base.
 */
export interface WorkflowStep {
  /** Unique identifier for this step (UUID4). */
  readonly step_id: string;
  /** Agent responsible for executing this step. */
  readonly agent_id: string;
  /** Action or operation name (e.g. "summarize", "translate"). */
  readonly action: string;
  /** Parameters passed to the agent for this action. */
  readonly params: Readonly<Record<string, unknown>>;
  /** Maximum seconds allowed for this step (null = no per-step timeout). */
  readonly timeout_seconds: number | null;
  /** step_id values that must complete before this step may execute. */
  readonly depends_on: readonly string[];
}

/**
 * Outcome of executing a single WorkflowStep.
 * Maps to StepResult dataclass in agent_mesh_router.workflows.base.
 */
export interface StepResult {
  /** Matches the WorkflowStep.step_id that produced this result. */
  readonly step_id: string;
  /** True when the step completed without error. */
  readonly success: boolean;
  /** Data returned by the agent for this step (null on failure). */
  readonly output: Readonly<Record<string, unknown>> | null;
  /** Wall-clock milliseconds taken to execute this step. */
  readonly duration_ms: number;
  /** Error description when success is false (null on success). */
  readonly error: string | null;
}

/**
 * Overall status of a workflow execution.
 * Maps to WorkflowStatus enum in agent_mesh_router.workflows.base.
 */
export type WorkflowStatus = "success" | "partial" | "failed" | "cancelled";

/**
 * Aggregate outcome of a complete workflow execution.
 * Maps to WorkflowResult dataclass in agent_mesh_router.workflows.base.
 */
export interface AgentRoute {
  /** Unique identifier for this workflow run (UUID4). */
  readonly workflow_id: string;
  /** Ordered list of individual step outcomes. */
  readonly step_results: readonly StepResult[];
  /** Overall workflow status. */
  readonly status: WorkflowStatus;
  /** Total wall-clock milliseconds from first step start to last step end. */
  readonly duration_ms: number;
}

// ---------------------------------------------------------------------------
// RouteTaskRequest
// ---------------------------------------------------------------------------

/**
 * Request payload for executing a multi-step workflow through the mesh.
 */
export interface RouteTaskRequest {
  /** Coordination pattern for this workflow. */
  readonly pattern: WorkflowPattern;
  /** Ordered list of steps to execute. */
  readonly steps: readonly Omit<WorkflowStep, "step_id">[];
  /** Optional workflow-level timeout in seconds. */
  readonly timeout_seconds?: number;
}

// ---------------------------------------------------------------------------
// Mesh topology
// ---------------------------------------------------------------------------

/**
 * Lifecycle status of an agent node in the fleet.
 * Maps to AgentStatus enum in agent_mesh_router.fleet.registry.
 */
export type AgentStatus = "HEALTHY" | "UNHEALTHY" | "DEGRADED" | "STARTING" | "STOPPING";

/**
 * A single agent node in the fleet registry.
 * Maps to AgentNode dataclass in agent_mesh_router.fleet.registry.
 */
export interface MeshNode {
  /** Unique agent identifier. */
  readonly agent_id: string;
  /** Capability strings this agent exposes. */
  readonly capabilities: readonly string[];
  /** Current lifecycle status. */
  readonly status: AgentStatus;
  /** Normalized load in [0.0, 1.0]; 0.0 = idle, 1.0 = fully loaded. */
  readonly load_score: number;
  /** Unix epoch seconds of the most recent heartbeat received. */
  readonly last_heartbeat: number;
  /** Free-form key-value metadata (region, version, model, etc.). */
  readonly metadata: Readonly<Record<string, string>>;
}

/**
 * Snapshot of the full mesh topology returned by getTopology().
 */
export interface MeshTopology {
  /** All currently registered agent nodes. */
  readonly nodes: readonly MeshNode[];
  /** Total number of registered nodes. */
  readonly total_count: number;
  /** Number of nodes with HEALTHY status. */
  readonly healthy_count: number;
  /** ISO-8601 UTC timestamp when this snapshot was captured. */
  readonly captured_at: string;
}

// ---------------------------------------------------------------------------
// Circuit breaker
// ---------------------------------------------------------------------------

/**
 * Possible states of a circuit breaker.
 * Maps to CircuitState enum in agent_mesh_router.resilience.circuit_breaker.
 */
export type CircuitBreakerState = "closed" | "open" | "half_open";

/**
 * Status snapshot for a single circuit breaker instance.
 * Maps to CircuitBreaker properties in agent_mesh_router.resilience.circuit_breaker.
 */
export interface CircuitBreakerStatus {
  /** Human-readable name for this circuit breaker. */
  readonly name: string;
  /** Current state of the circuit breaker. */
  readonly state: CircuitBreakerState;
  /** Consecutive failure count in the current CLOSED window. */
  readonly failure_count: number;
  /** Total calls attempted (not counting rejected calls). */
  readonly total_calls: number;
  /** Total calls rejected because the circuit was open. */
  readonly total_rejected: number;
  /** Number of failures required to open the circuit. */
  readonly failure_threshold: number;
  /** Seconds the circuit stays OPEN before transitioning to HALF_OPEN. */
  readonly recovery_timeout_seconds: number;
}

// ---------------------------------------------------------------------------
// Conflict resolution
// ---------------------------------------------------------------------------

/**
 * Request payload for resolving a conflict between competing agent outputs.
 */
export interface ResolveConflictRequest {
  /** Identifier of the workflow or task producing conflicting outputs. */
  readonly workflow_id: string;
  /** Competing outputs from different agents, keyed by agent_id. */
  readonly outputs: Readonly<Record<string, unknown>>;
  /** Resolution strategy hint (e.g. "majority", "weighted", "supervisor"). */
  readonly strategy?: string;
}

/**
 * Result of conflict resolution between competing agent outputs.
 */
export interface ConflictResolution {
  /** The workflow that produced conflicting outputs. */
  readonly workflow_id: string;
  /** The resolved output selected or synthesized from competing outputs. */
  readonly resolved_output: Readonly<Record<string, unknown>>;
  /** Agent ID whose output was selected (null for synthesized results). */
  readonly winning_agent_id: string | null;
  /** Resolution strategy that was applied. */
  readonly strategy_applied: string;
  /** Confidence score for the resolution in [0.0, 1.0]. */
  readonly confidence: number;
}

// ---------------------------------------------------------------------------
// Routing configuration
// ---------------------------------------------------------------------------

/**
 * Configuration for the mesh router.
 * Maps to router configuration options across agent_mesh_router.
 */
export interface RoutingConfig {
  /** Default workflow pattern to use when not specified. */
  readonly default_pattern: WorkflowPattern;
  /** Default message priority level. */
  readonly default_priority: MessagePriority;
  /** Maximum retries for failed message delivery. */
  readonly max_retries: number;
  /** Whether to enable circuit breaker protection on all routes. */
  readonly circuit_breaker_enabled: boolean;
  /** Seconds before an agent is considered stale (missed heartbeats). */
  readonly heartbeat_timeout_seconds: number;
}

// ---------------------------------------------------------------------------
// API result wrapper (shared pattern)
// ---------------------------------------------------------------------------

/** Standard error payload returned by the agent-mesh-router API. */
export interface ApiError {
  readonly error: string;
  readonly detail: string;
}

/** Result type for all client operations. */
export type ApiResult<T> =
  | { readonly ok: true; readonly data: T }
  | { readonly ok: false; readonly error: ApiError; readonly status: number };
