/**
 * @aumos/agent-mesh-router
 *
 * TypeScript client for the AumOS agent-mesh-router service.
 * Provides HTTP client and type definitions for multi-agent messaging,
 * workflow orchestration, circuit breaker inspection, and conflict resolution.
 */

// Client and configuration
export type { AgentMeshRouterClient, AgentMeshRouterClientConfig } from "./client.js";
export { createAgentMeshRouterClient } from "./client.js";

// Core types
export type {
  MessageType,
  MessagePriority,
  Message,
  SendMessageRequest,
  WorkflowPattern,
  WorkflowStep,
  StepResult,
  WorkflowStatus,
  AgentRoute,
  RouteTaskRequest,
  AgentStatus,
  MeshNode,
  MeshTopology,
  CircuitBreakerState,
  CircuitBreakerStatus,
  ResolveConflictRequest,
  ConflictResolution,
  RoutingConfig,
  ApiError,
  ApiResult,
} from "./types.js";
