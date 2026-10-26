export const AGENT_URL =
  process.env.NEXT_PUBLIC_AGENT_URL ?? "http://localhost:8001";

export type Role = "user" | "assistant";

export interface Source {
  text: string;
  source: string;
  heading: string;
  score: number;
}

export interface ToolCall {
  name: string;
  input: Record<string, unknown>;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
}

export type AgentEvent =
  | { type: "text"; text: string }
  | { type: "tool"; name: string; input: Record<string, unknown> }
  | { type: "sources"; sources: Source[] }
  | { type: "error"; message: string }
  | { type: "done"; turns: number; latency_ms: number; usage: Usage };

export interface Metrics {
  total_requests: number;
  total_errors: number;
  avg_latency_ms: number;
  avg_turns: number;
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  tool_calls: Record<string, number>;
}

/** POST the conversation and invoke onEvent for each Server-Sent Event. */
export async function streamChat(
  messages: { role: Role; content: string }[],
  onEvent: (e: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`${AGENT_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages }),
    signal,
  });
  if (!res.ok || !res.body) {
    throw new Error(`Agent returned ${res.status}`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";
    for (const frame of frames) {
      if (frame.startsWith("data: ")) onEvent(JSON.parse(frame.slice(6)));
    }
  }
}
