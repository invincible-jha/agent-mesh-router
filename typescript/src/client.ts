/**
 * HTTP client for the agent-mesh-router service API.
 *
 * Uses the Fetch API (available natively in Node 18+, browsers, and Deno).
 * No external dependencies required.
 *
 * @example
 * ```ts
 * import { createAgentMeshRouterClient } from "@aumos/agent-mesh-router";
 *
 * const client = createAgentMeshRouterClient({ baseUrl: "http://localhost:8080" });
 *
 * // Send a task message to an agent
 * const result = await client.sendMessage({
 *   recipient_id: "summarizer-agent",
 *   message_type: "task",
 *   payload: { text: "Summarize the quarterly report." },
 * });
 *
 * if (result.ok) {
 *   console.log("Message dispatched:", result.data.message_id);
 * }
 *
 * // Execute a sequential workflow
 * const workflow = await client.routeTask({
 *   pattern: "sequential",
 *   steps: [
 *     { agent_id: "reader", action: "extract", params: {}, timeout_seconds: 30, depends_on: [] },
 *     { agent_id: "writer", action: "summarize", params: {}, timeout_seconds: 60, depends_on: [] },
 *   ],
 * });
 * ```
 */

import type {
  AgentRoute,
  ApiError,
  ApiResult,
  CircuitBreakerStatus,
  ConflictResolution,
  Message,
  MeshTopology,
  ResolveConflictRequest,
  RouteTaskRequest,
  RoutingConfig,
  SendMessageRequest,
} from "./types.js";

// ---------------------------------------------------------------------------
// Client configuration
// ---------------------------------------------------------------------------

/** Configuration options for the AgentMeshRouterClient. */
export interface AgentMeshRouterClientConfig {
  /** Base URL of the agent-mesh-router server (e.g. "http://localhost:8080"). */
  readonly baseUrl: string;
  /** Optional request timeout in milliseconds (default: 30000). */
  readonly timeoutMs?: number;
  /** Optional extra HTTP headers sent with every request. */
  readonly headers?: Readonly<Record<string, string>>;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

async function fetchJson<T>(
  url: string,
  init: RequestInit,
  timeoutMs: number,
): Promise<ApiResult<T>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timeoutId);

    const body = await response.json() as unknown;

    if (!response.ok) {
      const errorBody = body as Partial<ApiError>;
      return {
        ok: false,
        error: {
          error: errorBody.error ?? "Unknown error",
          detail: errorBody.detail ?? "",
        },
        status: response.status,
      };
    }

    return { ok: true, data: body as T };
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    const message = err instanceof Error ? err.message : String(err);
    return {
      ok: false,
      error: { error: "Network error", detail: message },
      status: 0,
    };
  }
}

function buildHeaders(
  extraHeaders: Readonly<Record<string, string>> | undefined,
): Record<string, string> {
  return {
    "Content-Type": "application/json",
    Accept: "application/json",
    ...extraHeaders,
  };
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-mesh-router service. */
export interface AgentMeshRouterClient {
  /**
   * Dispatch a message to a target agent in the mesh.
   *
   * Supports all MessageType variants (task, query, broadcast, etc.).
   * Returns the created Message record including the assigned message_id.
   *
   * @param request - Message dispatch payload with recipient and type.
   * @returns The created Message record with message_id and metadata.
   */
  sendMessage(request: SendMessageRequest): Promise<ApiResult<Message>>;

  /**
   * Execute a multi-step workflow through the agent mesh.
   *
   * The workflow pattern controls how steps are coordinated:
   * - sequential: steps run one after another in order
   * - parallel: all steps run concurrently with fan-out/fan-in
   * - hierarchical: a supervisor agent delegates to sub-agents
   * - competitive: all steps race; first result wins
   * - consensus: results are collected and voted upon
   *
   * @param request - Workflow execution request with pattern and steps.
   * @returns AgentRoute with per-step results, status, and total duration.
   */
  routeTask(request: RouteTaskRequest): Promise<ApiResult<AgentRoute>>;

  /**
   * Retrieve the current mesh topology including all registered agent nodes.
   *
   * @returns MeshTopology snapshot with nodes, counts, and capture timestamp.
   */
  getTopology(): Promise<ApiResult<MeshTopology>>;

  /**
   * Retrieve the status of a named circuit breaker.
   *
   * Circuit breakers protect mesh routes from cascading failures. This
   * endpoint returns the current state (closed/open/half_open) and
   * cumulative call/rejection counts.
   *
   * @param circuitName - The human-readable circuit breaker name.
   * @returns CircuitBreakerStatus snapshot with state and counters.
   */
  getCircuitBreakerStatus(circuitName: string): Promise<ApiResult<CircuitBreakerStatus>>;

  /**
   * Resolve a conflict between competing agent outputs.
   *
   * Used after competitive or consensus workflows produce disagreeing results.
   * The configured resolution strategy (majority, weighted, supervisor, etc.)
   * determines how a winner is selected or synthesized.
   *
   * @param request - Conflict resolution request with competing outputs.
   * @returns ConflictResolution with the resolved output and winning agent.
   */
  resolveConflict(
    request: ResolveConflictRequest,
  ): Promise<ApiResult<ConflictResolution>>;
}

// ---------------------------------------------------------------------------
// Client factory
// ---------------------------------------------------------------------------

/**
 * Create a typed HTTP client for the agent-mesh-router service.
 *
 * @param config - Client configuration including base URL.
 * @returns An AgentMeshRouterClient instance.
 */
export function createAgentMeshRouterClient(
  config: AgentMeshRouterClientConfig,
): AgentMeshRouterClient {
  const { baseUrl, timeoutMs = 30_000, headers: extraHeaders } = config;
  const baseHeaders = buildHeaders(extraHeaders);

  return {
    async sendMessage(
      request: SendMessageRequest,
    ): Promise<ApiResult<Message>> {
      return fetchJson<Message>(
        `${baseUrl}/messages`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async routeTask(request: RouteTaskRequest): Promise<ApiResult<AgentRoute>> {
      return fetchJson<AgentRoute>(
        `${baseUrl}/workflows`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },

    async getTopology(): Promise<ApiResult<MeshTopology>> {
      return fetchJson<MeshTopology>(
        `${baseUrl}/topology`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async getCircuitBreakerStatus(
      circuitName: string,
    ): Promise<ApiResult<CircuitBreakerStatus>> {
      return fetchJson<CircuitBreakerStatus>(
        `${baseUrl}/circuit-breakers/${encodeURIComponent(circuitName)}`,
        { method: "GET", headers: baseHeaders },
        timeoutMs,
      );
    },

    async resolveConflict(
      request: ResolveConflictRequest,
    ): Promise<ApiResult<ConflictResolution>> {
      return fetchJson<ConflictResolution>(
        `${baseUrl}/conflicts/resolve`,
        {
          method: "POST",
          headers: baseHeaders,
          body: JSON.stringify(request),
        },
        timeoutMs,
      );
    },
  };
}

/** Re-export config and key types for convenience. */
export type {
  AgentRoute,
  CircuitBreakerStatus,
  ConflictResolution,
  Message,
  MeshTopology,
  ResolveConflictRequest,
  RouteTaskRequest,
  RoutingConfig,
  SendMessageRequest,
};
