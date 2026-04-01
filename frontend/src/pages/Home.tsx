import { useState, useEffect } from "react";
import { AnimatePresence } from "framer-motion";
import { Background } from "@/components/Background";
import { UploadView } from "@/components/UploadView";
import { ProcessingView } from "@/components/ProcessingView";
import { ResultsView } from "@/components/ResultsView";
import { useUploadAudioMutation, useJobStatusQuery } from "@/hooks/use-audio-process";
import { useToast } from "@/hooks/use-toast";

type ViewState = "upload" | "processing" | "results";

export default function Home() {
  const [view, setView] = useState<ViewState>("upload");
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const { toast } = useToast();

  const uploadMutation = useUploadAudioMutation(setUploadProgress);
  const { data: statusData, error: statusError } = useJobStatusQuery(jobId);

  useEffect(() => {
    if (!statusData) return;
    if (statusData.status === "completed" && view !== "results") {
      setView("results");
    }
  }, [statusData, view]);

  useEffect(() => {
    if (statusError) {
      toast({
        title: "Connection Error",
        description: "Lost connection to the processing server.",
        variant: "destructive",
      });
    }
  }, [statusError, toast]);

  const handleStartProcess = async (data: {
    audio?: File;
    youtube_url?: string;
    remove_fatiha: string;
  }) => {
    setUploadProgress(0);
    try {
      const res = await uploadMutation.mutateAsync(data);
      setJobId(res.jobId);
      setView("processing");
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to start processing job";
      toast({
        title: "Upload Failed",
        description: message,
        variant: "destructive",
      });
    }
  };

  const handleReset = () => {
    setView("upload");
    setJobId(null);
    setUploadProgress(0);
    uploadMutation.reset();
  };

  return (
    <div className="min-h-screen w-full relative overflow-hidden text-foreground">
      <Background />
      <main className="relative z-10 min-h-screen flex flex-col justify-end">
        <div className="w-full max-w-7xl mx-auto px-6 sm:px-10 lg:px-16 pb-16 pt-[55vh]">
          <AnimatePresence mode="wait">
            {view === "upload" && (
              <UploadView
                key="upload"
                onStart={handleStartProcess}
                isPending={uploadMutation.isPending}
                uploadProgress={uploadProgress}
              />
            )}
            {view === "processing" && (
              <ProcessingView
                key="processing"
                status={statusData}
                onReset={handleReset}
              />
            )}
            {view === "results" && statusData && (
              <ResultsView
                key="results"
                status={statusData}
                onReset={handleReset}
              />
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
