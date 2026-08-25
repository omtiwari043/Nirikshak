import { useEffect, useRef, useState, useCallback } from "react";
import api from "../api/client";

const LIVE_CHECK_INTERVAL_MS = 1300;
// Downscaled frame used only for the live feedback poll — keeps the upload
// small and the fast OCR pass quick. The final capture uses the full
// native resolution frame instead (see captureFullFrame).
const LIVE_CHECK_MAX_WIDTH = 640;

/**
 * Live camera capture with real-time framing/quality/OCR feedback.
 *
 * Props:
 *   onCapture(file: File, previewUrl: string) — called when the officer
 *     presses "Capture". The parent owns what happens next (preview + submit
 *     via the same flow as a regular file upload).
 */
export default function LiveCamera({ onCapture }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null); // full-res capture canvas
  const liveCanvasRef = useRef(null); // downscaled live-check canvas
  const streamRef = useRef(null);
  const pollTimerRef = useRef(null);
  const pollInFlightRef = useRef(false);

  const [facingMode, setFacingMode] = useState("environment"); // back camera by default on phones
  const [active, setActive] = useState(false);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [checking, setChecking] = useState(false);

  const stopStream = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setActive(false);
    setFeedback(null);
  }, []);

  const startStream = useCallback(async () => {
    setError("");
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not support camera access. Use file upload instead, or open this page over HTTPS.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setActive(true);
    } catch (err) {
      setError(
        err?.name === "NotAllowedError"
          ? "Camera permission was denied. Allow camera access and try again."
          : "Could not access the camera. It may be in use by another app, or unavailable on this device."
      );
    }
  }, [facingMode]);

  // Restart the stream whenever the officer switches front/back camera.
  useEffect(() => {
    if (active) {
      stopStream();
      startStream();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facingMode]);

  useEffect(() => stopStream, [stopStream]); // cleanup on unmount

  const runLiveCheck = useCallback(async () => {
    if (pollInFlightRef.current || !videoRef.current || videoRef.current.readyState < 2) return;
    const video = videoRef.current;
    const canvas = liveCanvasRef.current;
    if (!canvas || !video.videoWidth) return;

    const scale = LIVE_CHECK_MAX_WIDTH / video.videoWidth;
    canvas.width = LIVE_CHECK_MAX_WIDTH;
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      async (blob) => {
        if (!blob) return;
        pollInFlightRef.current = true;
        setChecking(true);
        try {
          const form = new FormData();
          form.append("frame", blob, "frame.jpg");
          const { data } = await api.post("/scans/live-check", form, {
            headers: { "Content-Type": "multipart/form-data" },
          });
          setFeedback(data);
        } catch {
          // Silently skip a failed live-check tick — the next poll will retry.
          // A hard OCR/auth error will still surface on the real capture+submit.
        } finally {
          pollInFlightRef.current = false;
          setChecking(false);
        }
      },
      "image/jpeg",
      0.7
    );
  }, []);

  useEffect(() => {
    if (!active) return;
    pollTimerRef.current = setInterval(runLiveCheck, LIVE_CHECK_INTERVAL_MS);
    runLiveCheck(); // fire one immediately instead of waiting for the first interval tick
    return () => clearInterval(pollTimerRef.current);
  }, [active, runLiveCheck]);

  const handleCapture = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || !video.videoWidth) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const file = new File([blob], `live-capture-${Date.now()}.jpg`, { type: "image/jpeg" });
        onCapture(file, URL.createObjectURL(blob));
        stopStream();
      },
      "image/jpeg",
      0.95
    );
  };

  return (
    <div className="space-y-3">
      {!active ? (
        <button type="button" onClick={startStream} className="btn-secondary w-full">
          📷 Start live camera
        </button>
      ) : (
        <>
          <div className="relative rounded-lg overflow-hidden bg-black">
            {/* eslint-disable-next-line jsx-a11y/media-has-caption */}
            <video ref={videoRef} className="w-full max-h-80 object-contain" muted playsInline />

            {/* Framing guide — helps officers center the declaration panel */}
            <div className="pointer-events-none absolute inset-4 border-2 border-dashed border-white/70 rounded-lg" />

            <div className="absolute top-2 left-2 right-2 flex items-center justify-between gap-2">
              <span
                className={`text-xs font-medium px-2 py-1 rounded-full ${
                  feedback?.ready_to_capture
                    ? "bg-green-600 text-white"
                    : feedback
                    ? "bg-amber-500 text-white"
                    : "bg-black/50 text-white"
                }`}
              >
                {checking && !feedback
                  ? "Analyzing…"
                  : feedback?.ready_to_capture
                  ? "✓ Looks good — capture now"
                  : feedback
                  ? feedback.warnings[0] || "Keep steady…"
                  : "Point at the label"}
              </span>
              <button
                type="button"
                onClick={() => setFacingMode((m) => (m === "environment" ? "user" : "environment"))}
                className="text-xs bg-black/50 text-white px-2 py-1 rounded-full"
                title="Switch camera"
              >
                🔄 Flip
              </button>
            </div>

            {feedback && (
              <div className="absolute bottom-2 left-2 right-2 text-[11px] text-white bg-black/50 rounded-lg px-2 py-1">
                {feedback.lines_detected} text line(s) detected
                {feedback.keywords_detected?.length > 0 && ` • found: ${feedback.keywords_detected.join(", ")}`}
              </div>
            )}
          </div>

          <div className="flex gap-2">
            <button type="button" onClick={handleCapture} className="btn-primary flex-1">
              Capture
            </button>
            <button type="button" onClick={stopStream} className="btn-secondary">
              Cancel
            </button>
          </div>
        </>
      )}

      {error && <p className="text-sm text-red-600">{error}</p>}

      {/* Hidden canvases used only for frame capture, never rendered visibly */}
      <canvas ref={canvasRef} className="hidden" />
      <canvas ref={liveCanvasRef} className="hidden" />
    </div>
  );
}
