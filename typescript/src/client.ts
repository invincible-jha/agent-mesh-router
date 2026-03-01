/**
 * HTTP client for the agent-mesh-router service API.
 *
 * Delegates all HTTP transport to `@aumos/sdk-core` which provides
 * automatic retry with exponential back-off, timeout management via
 * `AbortSignal.timeout`, interceptor support, and a typed error hierarchy.
 *
 * The public-facing `ApiResult<T>` envelope is preserved for full
 * backward compatibility with existing callers.
 *
 * @example
 * ```ts
 * import { createAgentMeshRouterClient } from "@aumos/agent-mesh-router";
 *
 * const client = createAgentMeshRouterClient({ baseUrl: "http://localhost:8080" });
 *
 * const result = await client.sendMessage({
 *   recipient_id: "summarizer-agent",
 *   message_type: "task",
 *   payload: { text: "Summarize the quarterly report." },
 * });
 *
 * if (result.ok) {
 *   console.log("Message dispatched:", result.data.message_id);
 * }
 * ```
 */

import {
  createHttpClient,
  HttpError,
  NetworkError,
  TimeoutError,
  AumosError,
  type HttpClient,
} from "@aumos/sdk-core";

import type {
  AgentRoute,
  ApiResult,
  CircuitBreakerStatus,
  ConflictResolution,
  Message,
  MeshTopology,
  ResolveConflictRequest,
  RouteTaskRequest,
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
// Internal adapter
// ---------------------------------------------------------------------------

async function callApi<T>(
  operation: () => Promise<{ readonly data: T; readonly status: number }>,
): Promise<ApiResult<T>> {
  try {
    const response = await operation();
    return { ok: true, data: response.data };
  } catch (error: unknown) {
    if (error instanceof HttpError) {
      return {
        ok: false,
        error: { error: error.message, detail: String(error.body ?? "") },
        status: error.statusCode,
      };
    }
    if (error instanceof TimeoutError) {
      return {
        ok: false,
        error: { error: "Request timed out", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof NetworkError) {
      return {
        ok: false,
        error: { error: "Network error", detail: error.message },
        status: 0,
      };
    }
    if (error instanceof AumosError) {
      return {
        ok: false,
        error: { error: error.code, detail: error.message },
        status: error.statusCode ?? 0,
      };
    }
    const message = error instanceof Error ? error.message : String(error);
    return {
      ok: false,
      error: { error: "Unexpected error", detail: message },
      status: 0,
    };
  }
}

// ---------------------------------------------------------------------------
// Client interface
// ---------------------------------------------------------------------------

/** Typed HTTP client for the agent-mesh-router service. */
export interface AgentMeshRouterClient {
  /**
   * Dispatch a message to a target agent in the mesh.
   *
   * @param request - Message dispatch payload with recipient and type.
   * @returns The created Message record with message_id and metadata.
   */
  sendMessage(request: SendMessageRequest): Promise<ApiResult<Message>>;

  /**
   * Execute a multi-step workflow through the agent mesh.
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
   * @param circuitName - The human-readable circuit breaker name.
   * @returns CircuitBreakerStatus snapshot with state and counters.
   */
  getCircuitBreakerStatus(circuitName: string): Promise<ApiResult<CircuitBreakerStatus>>;

  /**
   * Resolve a conflict between competing agent outputs.
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
  const http: HttpClient = createHttpClient({
    baseUrl: config.baseUrl,
    timeout: config.timeoutMs ?? 30_000,
    defaultHeaders: config.headers,
  });

  return {
    sendMessage(request: SendMessageRequest): Promise<ApiResult<Message>> {
      return callApi(() => http.post<Message>("/messages", request));
    },

    routeTask(request: RouteTaskRequest): Promise<ApiResult<AgentRoute>> {
      return callApi(() => http.post<AgentRoute>("/workflows", request));
    },

    getTopology(): Promise<ApiResult<MeshTopology>> {
      return callApi(() => http.get<MeshTopology>("/topology"));
    },

    getCircuitBreakerStatus(
      circuitName: string,
    ): Promise<ApiResult<CircuitBreakerStatus>> {
      return callApi(() =>
        http.get<CircuitBreakerStatus>(
          `/circuit-breakers/${encodeURIComponent(circuitName)}`,
        ),
      );
    },

    resolveConflict(request: ResolveConflictRequest): Promise<ApiResult<ConflictResolution>> {
      return callApi(() =>
        http.post<ConflictResolution>("/conflicts/resolve", request),
      );
    },
  };
}
