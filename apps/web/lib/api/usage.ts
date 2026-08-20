import { get } from "@/lib/api/client";

export interface UsageMetric {
  period: string;
  requests: number;
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  latencyMs: number;
  errors: number;
}

export interface UsageSummary {
  requests: number;
  tokens: number;
  cost: number;
  errorRate: number;
  breakdown: Record<string, UsageMetric>;
}

export interface UsageApi {
  summary(range?: string): Promise<UsageSummary>;
  timeSeries(range?: string, granularity?: string): Promise<UsageMetric[]>;
  byModel(): Promise<Record<string, UsageMetric>>;
}

export const usageApi: UsageApi = {
  summary: (range = "30d") => get<UsageSummary>(`/v1/usage/summary?range=${range}`),
  timeSeries: (range = "30d", granularity = "day") =>
    get<UsageMetric[]>(
      `/v1/usage/timeseries?range=${range}&granularity=${granularity}`
    ),
  byModel: () => get<Record<string, UsageMetric>>("/v1/usage/by-model"),
};