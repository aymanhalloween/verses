import { motion, AnimatePresence } from "framer-motion";
import { AlertCircle, RotateCcw } from "lucide-react";
import type { JobStatus } from "@/hooks/use-audio-process";

interface ProcessingViewProps {
  status: JobStatus | undefined;
  onReset: () => void;
}

const STAGES: { key: string; label: string; sublabel: string }[] = [
  { key: "downloading", label: "Downloading", sublabel: "Fetching audio from source" },
  { key: "converting", label: "Converting", sublabel: "Preparing audio for analysis" },
  { key: "transcribing", label: "Transcribing", sublabel: "Reading the recitation" },
  { key: "filtering", label: "Filtering", sublabel: "Removing takbeers, dhikr, and silence" },
  { key: "stitching", label: "Assembling", sublabel: "Building the clean recording" },
];

export function ProcessingView({ status, onReset }: ProcessingViewProps) {
  const isFailed = status?.status === "failed";
  const progressPercent = (status?.progress || 0) * 100;

  let currentStageIndex = 0;
  if (status?.stage) {
    const foundIdx = STAGES.findIndex(
      (s) => s.key === status.stage?.toLowerCase(),
    );
    if (foundIdx !== -1) currentStageIndex = foundIdx;
  } else {
    if (progressPercent > 88) currentStageIndex = 4;
    else if (progressPercent > 45) currentStageIndex = 3;
    else if (progressPercent > 25) currentStageIndex = 2;
    else if (progressPercent > 15) currentStageIndex = 1;
  }

  const currentStage = STAGES[currentStageIndex];

  if (isFailed) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="w-full max-w-lg mx-auto flex flex-col relative z-10"
      >
        <p className="label-caps mb-14">Processing failed</p>
        <div className="flex items-start gap-5 mb-14">
          <div className="w-10 h-10 rounded-full bg-red-500/10 flex items-center justify-center shrink-0 mt-1">
            <AlertCircle className="w-5 h-5 text-red-400" />
          </div>
          <div>
            <h2 className="text-3xl font-display font-light text-white mb-2">
              Something went wrong
            </h2>
            <p className="text-sm text-zinc-500 font-light leading-relaxed">
              {status?.error || "An error occurred during processing. Please try again."}
            </p>
          </div>
        </div>
        <button
          onClick={onReset}
          className="self-start flex items-center gap-3 text-[11px] uppercase tracking-[0.2em] font-semibold text-zinc-400 hover:text-white transition-colors"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          Try Again
        </button>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.9, ease: "easeInOut" }}
      className="w-full max-w-lg mx-auto flex flex-col relative z-10"
    >
      <AnimatePresence mode="wait">
        <motion.div
          key={currentStageIndex}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
        >
          <p className="label-caps mb-14">
            Step {currentStageIndex + 1} of {STAGES.length} · {currentStage?.label.toUpperCase()}
          </p>
          <h2 className="text-[2.6rem] md:text-[3.2rem] font-display font-light text-white leading-[1.1] tracking-tight mb-5">
            {currentStage?.label}
          </h2>
          <p className="text-sm text-zinc-500 font-light mb-16">
            {currentStage?.sublabel}
          </p>
        </motion.div>
      </AnimatePresence>

      <div className="flex flex-col gap-3">
        <div className="h-px bg-white/8 w-full overflow-hidden rounded-full">
          <motion.div
            className="h-full bg-white"
            initial={{ width: "0%" }}
            animate={{ width: `${progressPercent}%` }}
            transition={{ duration: 0.8, ease: "easeOut" }}
          />
        </div>
        <div className="flex justify-between">
          <span className="label-caps text-zinc-700">Processing</span>
          <span className="label-caps text-zinc-600">{Math.round(progressPercent)}%</span>
        </div>
      </div>

      <div className="mt-16 flex flex-col gap-2">
        {STAGES.map((stage, i) => (
          <div key={stage.key} className="flex items-center gap-3">
            <div className={`w-1.5 h-1.5 rounded-full transition-colors duration-500 ${
              i < currentStageIndex
                ? "bg-white/40"
                : i === currentStageIndex
                ? "bg-white"
                : "bg-white/10"
            }`} />
            <span className={`text-xs transition-colors duration-500 ${
              i < currentStageIndex
                ? "text-zinc-600"
                : i === currentStageIndex
                ? "text-zinc-300"
                : "text-zinc-700"
            }`}>
              {stage.label}
            </span>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
