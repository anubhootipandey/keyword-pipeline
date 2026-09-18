/**
 * Minimal API client.
 *
 * Phase 1 only needs a health check to prove the frontend can talk to
 * the backend. The real `/api/analyze` call gets added in Phase 11/12
 * once the backend endpoint actually exists.
 */

const API_BASE_URL: string = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface HealthResponse {
  status: string;
  app_name: string;
  environment: string;
}

export async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch(`${API_BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error(`Health check failed with status ${response.status}`);
  }

  return response.json();
}
