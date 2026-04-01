import { motion } from "framer-motion";
import { ArrowDown, RotateCcw } from "lucide-react";
import type { JobStatus } from "@/hooks/use-audio-process";

interface ResultsViewProps {
  status: JobStatus;
  onReset: () => void;
}

function formatDuration(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = Math.round(secs % 60);
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function ResultsView({ status, onReset }: ResultsViewProps) {
  const summary = status.summary;
  const durationSecs = summary?.durationSecs ?? 0;
  const removedSecs = summary?.removedSecs ?? 0;
  const segmentsRemoved = summary?.segmentsRemoved ?? 0;

  const handleDownload = () => {
    window.location.href = `/api/download/${status.jobId}`;
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.9, ease: "easeOut" }}
      className="w-full max-w-lg mx-auto flex flex-col relative z-10"
    >
      <p className="label-caps mb-14">Extraction complete</p>

      <h1 className="text-[2.6rem] md:text-[3.2rem] font-display font-light text-white leading-[1.1] tracking-tight mb-6">
        Your recitation<br />is ready.
      </h1>

      <p className="text-sm text-zinc-500 font-light mb-10 leading-relaxed">
        Takbeers, dhikr, and silence have been removed. The clean recitation is ready to download.
      </p>

      <div className="grid grid-cols-3 gap-px bg-white/8 rounded-2xl overflow-hidden mb-12">
        {[
          { value: formatDuration(durationSecs), label: "Clean Audio" },
          { value: formatDuration(removedSecs), label: "Removed" },
          { value: segmentsRemoved, label: segmentsRemoved === 1 ? "Segment" : "Segments" },
        ].map(({ value, label }) => (
          <div key={label} className="bg-[#0D0D0F] px-5 py-6 flex flex-col gap-1.5">
            <span className="text-2xl font-display font-light text-white">{value}</span>
            <span className="label-caps">{label}</span>
          </div>
        ))}
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <button
          onClick={handleDownload}
          className="flex-1 h-13 rounded-full bg-white text-[#0D0D0F] text-[11px] uppercase tracking-[0.2em] font-semibold hover:bg-zinc-100 transition-colors flex items-center justify-center gap-2.5"
        >
          <ArrowDown className="w-3.5 h-3.5" />
          Download Recitation
        </button>
        <button
          onClick={onReset}
          className="flex-1 h-13 rounded-full border border-white/15 text-zinc-400 text-[11px] uppercase tracking-[0.2em] font-semibold hover:bg-white/5 hover:text-white transition-colors flex items-center justify-center gap-2.5"
        >
          <RotateCcw className="w-3 h-3" />
          Process Another
        </button>
      </div>
    </motion.div>
  );
}
