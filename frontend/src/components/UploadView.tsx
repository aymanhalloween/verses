import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import { FileAudio, X, ArrowRight, Loader2 } from "lucide-react";

interface UploadViewProps {
  onStart: (data: { audio?: File; youtube_url?: string; remove_fatiha: string }) => void;
  isPending: boolean;
  uploadProgress?: number;
}

export function UploadView({ onStart, isPending, uploadProgress = 0 }: UploadViewProps) {
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [removeFatiha, setRemoveFatiha] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setFile(acceptedFiles[0]);
      setYoutubeUrl("");
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "audio/mpeg": [".mp3"],
      "audio/wav": [".wav"],
      "audio/mp4": [".m4a"],
      "audio/ogg": [".ogg"],
    },
    maxFiles: 1,
  });

  const handleSubmit = () => {
    if (!file && !youtubeUrl.trim()) {
      setError("Provide an audio file or a YouTube URL.");
      return;
    }
    if (youtubeUrl && !youtubeUrl.includes("youtube.com") && !youtubeUrl.includes("youtu.be")) {
      setError("Please enter a valid YouTube URL.");
      return;
    }
    onStart({
      audio: file || undefined,
      youtube_url: youtubeUrl.trim() || undefined,
      remove_fatiha: removeFatiha ? "true" : "false",
    });
  };

  const hasInput = !!file || !!youtubeUrl.trim();

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
      className="w-full max-w-lg mx-auto flex flex-col gap-0 relative z-10"
    >
      <div className="mb-14">
        <p className="label-caps mb-5">Taraweeh · Audio Extractor</p>
        <h1 className="text-[2.6rem] md:text-[3.2rem] font-display font-light text-white leading-[1.1] tracking-tight">
          Strip the in-between.<br />Keep the Quran.
        </h1>
        <p className="mt-5 text-[0.9rem] text-zinc-400 font-light leading-relaxed max-w-sm">
          Upload any salah recording and get back a clean MP3 of just the Quranic recitation — takbeers, dhikr, and silence removed automatically.
        </p>
      </div>

      <div className="flex flex-col gap-10">
        <div className="flex flex-col gap-3">
          <p className="label-caps">Drop your recording</p>
          <AnimatePresence mode="wait">
            {file ? (
              <motion.div
                key="file"
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                className="flex items-center justify-between py-4 border-b border-white/10"
              >
                <div className="flex items-center gap-4 overflow-hidden">
                  <div className="w-9 h-9 rounded-full bg-white/8 flex items-center justify-center shrink-0">
                    <FileAudio className="w-4 h-4 text-zinc-300" />
                  </div>
                  <div className="flex flex-col truncate">
                    <span className="text-sm text-zinc-200 truncate">{file.name}</span>
                    <span className="text-[11px] text-zinc-500 mt-0.5">{(file.size / (1024 * 1024)).toFixed(1)} MB</span>
                  </div>
                </div>
                <button
                  onClick={() => setFile(null)}
                  className="p-1.5 rounded-full hover:bg-white/8 text-zinc-500 hover:text-zinc-200 transition-colors"
                >
                  <X className="w-3.5 h-3.5" />
                </button>
              </motion.div>
            ) : (
              <motion.div
                key="dropzone"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              >
                <div
                  {...getRootProps()}
                  className={`
                    border rounded-2xl p-8 flex flex-col items-center justify-center gap-3 cursor-pointer transition-all duration-300
                    ${isDragActive
                      ? "border-white/30 bg-white/4"
                      : "border-white/8 bg-white/[0.015] hover:border-white/15 hover:bg-white/[0.025]"
                    }
                    ${youtubeUrl ? "opacity-30 pointer-events-none" : ""}
                  `}
                >
                  <input {...getInputProps()} disabled={!!youtubeUrl} />
                  <p className="text-sm text-zinc-300 font-light">
                    {isDragActive ? "Release to upload" : "Drag & drop MP3, WAV, or M4A"}
                  </p>
                  <p className="text-[11px] text-zinc-600 uppercase tracking-wider">or click to browse</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <div className="flex items-center gap-4">
          <div className="flex-1 h-px bg-white/8" />
          <span className="label-caps text-zinc-700">or</span>
          <div className="flex-1 h-px bg-white/8" />
        </div>

        <div className="flex flex-col gap-3">
          <p className="label-caps">YouTube link</p>
          <input
            type="text"
            placeholder="https://youtu.be/..."
            value={youtubeUrl}
            onChange={(e) => {
              setYoutubeUrl(e.target.value);
              if (e.target.value) setFile(null);
              setError(null);
            }}
            disabled={!!file}
            className={`
              w-full bg-transparent border-0 border-b py-3 text-sm text-white placeholder:text-white/20 outline-none transition-colors
              ${file ? "opacity-30 pointer-events-none border-white/8" : "border-white/15 hover:border-white/25 focus:border-white/40"}
            `}
          />
        </div>

        <div className="flex items-center justify-between py-1">
          <div className="flex flex-col gap-0.5">
            <p className="text-sm text-zinc-300 font-light">Remove Surah Al-Fatiha</p>
            <p className="text-[11px] text-zinc-600">Filter repetitive opening recitations</p>
          </div>
          <button
            onClick={() => setRemoveFatiha(!removeFatiha)}
            className={`relative w-11 h-6 rounded-full transition-colors duration-200 focus:outline-none ${
              removeFatiha ? "bg-white" : "bg-white/15"
            }`}
          >
            <span
              className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full transition-transform duration-200 ${
                removeFatiha ? "translate-x-5 bg-[#0D0D0F]" : "translate-x-0 bg-white/60"
              }`}
            />
          </button>
        </div>

        <AnimatePresence>
          {error && (
            <motion.p
              initial={{ opacity: 0, y: -4 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="text-red-400 text-xs tracking-wide"
            >
              {error}
            </motion.p>
          )}
        </AnimatePresence>

        <button
          onClick={handleSubmit}
          disabled={isPending || !hasInput}
          className={`
            w-full h-13 rounded-full text-[11px] uppercase tracking-[0.2em] font-semibold transition-all duration-300 flex items-center justify-center gap-3
            ${hasInput && !isPending
              ? "bg-white text-[#0D0D0F] hover:bg-zinc-100"
              : "bg-white/8 text-zinc-500 cursor-not-allowed"
            }
          `}
        >
          {isPending ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              {uploadProgress > 0 && uploadProgress < 100
                ? `Uploading ${uploadProgress}%`
                : "Starting"}
            </>
          ) : (
            <>
              Extract Recitation
              <ArrowRight className="w-3.5 h-3.5" />
            </>
          )}
        </button>
      </div>
    </motion.div>
  );
}
