"use client";

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { AgentEvent, Source, ToolCall, Usage, streamChat } from "@/lib/agent";

interface Turn {
  role: "user" | "assistant";
  content: string;
  tools?: ToolCall[];
  sources?: Source[];
  stats?: { turns: number; latency_ms: number; usage: Usage };
  error?: string;
}

const SUGGESTIONS = [
  "How does drift detection work?",
  "Is the deployed model drifting right now?",
  "How is traffic split in the current A/B experiment?",
  "Predict the price for a 6-room house in San Francisco built 40 years ago.",
];

export default function Chat() {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  function updateLast(fn: (t: Turn) => Turn) {
    setTurns((prev) => [...prev.slice(0, -1), fn(prev[prev.length - 1])]);
  }

  function apply(e: AgentEvent) {
    updateLast((t) => {
      switch (e.type) {
        case "text":
          return { ...t, content: t.content + e.text };
        case "tool":
          return { ...t, tools: [...(t.tools ?? []), { name: e.name, input: e.input }] };
        case "sources":
          return { ...t, sources: [...(t.sources ?? []), ...e.sources] };
        case "error":
          return { ...t, error: e.message };
        case "done":
          return { ...t, stats: { turns: e.turns, latency_ms: e.latency_ms, usage: e.usage } };
      }
    });
  }

  async function send(question: string) {
    if (!question.trim() || busy) return;
    const history = [...turns, { role: "user" as const, content: question }];
    setTurns([...history, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    try {
      const messages = history
        .filter((t) => t.content)
        .map(({ role, content }) => ({ role, content }));
      await streamChat(messages, apply);
    } catch (err) {
      updateLast((t) => ({ ...t, error: String(err) }));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-1 space-y-6 overflow-y-auto p-6">
        {turns.length === 0 && (
          <div className="mx-auto mt-16 max-w-xl text-center">
            <h2 className="text-2xl font-semibold">Ask the platform anything</h2>
            <p className="mt-2 text-sm text-zinc-400">
              Answers are grounded in the repo&apos;s docs and the live model API.
            </p>
            <div className="mt-8 grid gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-lg border border-zinc-800 p-3 text-left text-sm text-zinc-300 hover:border-emerald-600 hover:bg-zinc-900"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {turns.map((t, i) =>
          t.role === "user" ? (
            <div key={i} className="ml-auto max-w-2xl rounded-2xl bg-emerald-700/30 px-4 py-2">
              {t.content}
            </div>
          ) : (
            <AssistantTurn key={i} turn={t} pending={busy && i === turns.length - 1} />
          ),
        )}
        <div ref={bottom} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
        className="flex gap-2 border-t border-zinc-800 p-4"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about architecture, drift, A/B tests, predictions…"
          className="flex-1 rounded-lg border border-zinc-800 bg-zinc-900 px-4 py-2 outline-none focus:border-emerald-600"
        />
        <button
          disabled={busy || !input.trim()}
          className="rounded-lg bg-emerald-600 px-4 py-2 font-medium disabled:opacity-40"
        >
          Send
        </button>
      </form>
    </div>
  );
}

function AssistantTurn({ turn, pending }: { turn: Turn; pending: boolean }) {
  const sources = dedupe(turn.sources ?? []);
  return (
    <div className="max-w-3xl space-y-3">
      {turn.tools && (
        <div className="flex flex-wrap gap-2">
          {turn.tools.map((tool, i) => (
            <span
              key={i}
              title={JSON.stringify(tool.input)}
              className="rounded-full border border-zinc-700 px-2 py-0.5 font-mono text-xs text-zinc-400"
            >
              ⚙ {tool.name}
            </span>
          ))}
        </div>
      )}

      <div className="markdown">
        {turn.content ? (
          <ReactMarkdown>{turn.content}</ReactMarkdown>
        ) : (
          pending && <span className="animate-pulse text-zinc-500">Thinking…</span>
        )}
      </div>

      {turn.error && <p className="text-sm text-red-400">{turn.error}</p>}

      {sources.length > 0 && (
        <details className="text-sm text-zinc-400">
          <summary className="cursor-pointer">{sources.length} sources</summary>
          <ul className="mt-2 space-y-1">
            {sources.map((s) => (
              <li key={`${s.source}#${s.heading}`} className="font-mono text-xs">
                {s.source} › {s.heading || "(top)"}{" "}
                <span className="text-zinc-600">{s.score.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {turn.stats && (
        <p className="font-mono text-xs text-zinc-600">
          {turn.stats.latency_ms} ms · {turn.stats.turns} turns ·{" "}
          {turn.stats.usage.input_tokens} in / {turn.stats.usage.output_tokens} out ·{" "}
          {turn.stats.usage.cache_read_input_tokens} cached
        </p>
      )}
    </div>
  );
}

function dedupe(sources: Source[]): Source[] {
  const seen = new Map<string, Source>();
  for (const s of sources) seen.set(`${s.source}#${s.heading}`, s);
  return [...seen.values()];
}
