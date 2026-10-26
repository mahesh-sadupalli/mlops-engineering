"use client";

import { useEffect, useState } from "react";
import { AGENT_URL, Metrics } from "@/lib/agent";

export default function MetricsPanel() {
  const [m, setM] = useState<Metrics | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    const poll = () =>
      fetch(`${AGENT_URL}/metrics`)
        .then((r) => r.json())
        .then((data) => {
          setM(data);
          setDown(false);
        })
        .catch(() => setDown(true));
    poll();
    const id = setInterval(poll, 5000);
    return () => clearInterval(id);
  }, []);

  const cacheRate =
    m && m.input_tokens + m.cache_read_input_tokens > 0
      ? m.cache_read_input_tokens / (m.input_tokens + m.cache_read_input_tokens)
      : 0;

  return (
    <aside className="space-y-6 p-6 text-sm">
      <div className="flex items-center gap-2">
        <span className={`h-2 w-2 rounded-full ${down ? "bg-red-500" : "bg-emerald-500"}`} />
        <span className="text-zinc-400">{down ? "Agent offline" : "Agent online"}</span>
      </div>

      {m && (
        <>
          <dl className="grid grid-cols-2 gap-4">
            <Stat label="Requests" value={m.total_requests} />
            <Stat label="Errors" value={m.total_errors} />
            <Stat label="Avg latency" value={`${Math.round(m.avg_latency_ms)} ms`} />
            <Stat label="Avg turns" value={m.avg_turns} />
            <Stat label="Output tokens" value={m.output_tokens.toLocaleString()} />
            <Stat label="Cache hit" value={`${Math.round(cacheRate * 100)}%`} />
          </dl>

          <div>
            <h3 className="mb-2 text-xs uppercase tracking-wide text-zinc-500">Tool calls</h3>
            {Object.keys(m.tool_calls).length === 0 ? (
              <p className="text-zinc-600">None yet</p>
            ) : (
              <ul className="space-y-1 font-mono text-xs">
                {Object.entries(m.tool_calls).map(([name, n]) => (
                  <li key={name} className="flex justify-between">
                    <span className="text-zinc-400">{name}</span>
                    <span>{n}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </>
      )}
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-zinc-500">{label}</dt>
      <dd className="mt-1 font-mono text-lg">{value}</dd>
    </div>
  );
}
