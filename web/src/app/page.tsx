import Chat from "@/components/Chat";
import MetricsPanel from "@/components/MetricsPanel";

export default function Home() {
  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-zinc-800 px-6 py-3">
        <h1 className="font-semibold">
          MLOps Copilot{" "}
          <span className="font-normal text-zinc-500">· house-prices platform</span>
        </h1>
        <a
          href="https://github.com/mahesh-sadupalli/mlops-engineering"
          className="text-sm text-zinc-400 hover:text-zinc-200"
        >
          GitHub
        </a>
      </header>
      <main className="grid min-h-0 flex-1 md:grid-cols-[1fr_280px]">
        <Chat />
        <div className="hidden border-l border-zinc-800 md:block">
          <MetricsPanel />
        </div>
      </main>
    </div>
  );
}
