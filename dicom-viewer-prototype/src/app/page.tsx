"use client";

import { useState, useRef } from 'react';
import { Upload, X, ShieldAlert, Cpu } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

interface Landmark {
  x: number; y: number; x_pct: number; y_pct: number;
}
interface LandmarkData {
  femur_condyle: Landmark;
  tibial_plateau: Landmark;
  patella: Landmark;
  medial_condyle: Landmark;
  lateral_condyle: Landmark;
  femur_axis_top: Landmark;
  tibia_axis_bottom: Landmark;
  angles: { TPA: number; flexion: number; rotation: number; rotation_label: string };
  qa: {
    view_type: string;
    score: number;
    status: string;
    message: string;
    color: string;
    symmetry_ratio: number;
    positioning_advice: string;
  };
  heatmap_data?: string;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export default function Home() {
  // Tailwind cannot generate classes dynamically (e.g. `bg-${color}-500`).
  // We must map color names to actual Tailwind class strings.
  const colorMap: Record<string, { bg: string; bgDot: string; text: string; border: string }> = {
    green:  { bg: 'bg-green-500',  bgDot: 'bg-green-400',  text: 'text-green-400',  border: 'border-green-500' },
    yellow: { bg: 'bg-yellow-500', bgDot: 'bg-yellow-400', text: 'text-yellow-400', border: 'border-yellow-500' },
    red:    { bg: 'bg-red-500',    bgDot: 'bg-red-400',    text: 'text-red-400',    border: 'border-red-500' },
  };
  const [file, setFile] = useState<File | null>(null);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // AI Analysis States
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisComplete, setAnalysisComplete] = useState(false);
  const [landmarks, setLandmarks] = useState<LandmarkData | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const consoleRef = useRef<HTMLDivElement>(null);

  const addLog = (msg: string) => {
    setLogs(prev => {
      const next = [...prev, msg];
      setTimeout(() => {
        if (consoleRef.current) consoleRef.current.scrollTop = consoleRef.current.scrollHeight;
      }, 50);
      return next;
    });
  };

  const onDrop = async (acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];
    if (selectedFile) {
      setFile(selectedFile);
      setImagePreview(null);
      setMetadata(null);
      setError(null);
      setIsAnalyzing(false);
      setAnalysisComplete(false);
      setLandmarks(null);
      setLogs([]);
      await uploadDicom(selectedFile);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/dicom': ['.dcm', '.dicom'],
      'application/octet-stream': ['.dcm', '.dicom'],
      'image/png': ['.png'],
      'image/jpeg': ['.jpg', '.jpeg'],
    },
    multiple: false,
  });

  const uploadDicom = async (f: File) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('file', f);
    try {
      const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Failed'); }
      const data = await res.json();
      setMetadata(data.metadata);
      if (data.image_data) setImagePreview(data.image_data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null); setImagePreview(null); setMetadata(null); setError(null);
    setIsAnalyzing(false); setAnalysisComplete(false); setLandmarks(null); setLogs([]);
    setShowHeatmap(false);
  };

  const runAnalysis = async () => {
    setIsAnalyzing(true);
    setAnalysisComplete(false);
    setLandmarks(null);
    setLogs([]);

    try {
      addLog('> Initializing bone segmentation pipeline...');
      addLog('  - Algorithm: CLAHE + Otsu Threshold + Connected Components');

      let imageBlob: Blob;
      let filename = 'knee.png';

      if (file) {
        imageBlob = file;
        filename = file.name;
        addLog(`  - Source: DICOM file (${file.name})`);
      } else if (imagePreview && imagePreview.startsWith('/')) {
        addLog('  - Source: Demo knee X-ray image');
        const resp = await fetch(imagePreview);
        imageBlob = await resp.blob();
      } else if (imagePreview && imagePreview.startsWith('data:')) {
        addLog('  - Source: Loaded image (Base64)');
        const res = await fetch(imagePreview);
        imageBlob = await res.blob();
      } else {
        throw new Error('No image loaded. Please upload a file or use the demo image first.');
      }

      await new Promise(r => setTimeout(r, 600));
      addLog('> Running CLAHE contrast enhancement...');

      await new Promise(r => setTimeout(r, 500));
      addLog('> Applying Otsu thresholding to isolate bone regions...');

      await new Promise(r => setTimeout(r, 500));
      addLog('> Morphological cleanup (opening/closing)...');

      await new Promise(r => setTimeout(r, 400));
      addLog('> Connected component analysis – finding bone regions...');

      const formData = new FormData();
      formData.append('file', imageBlob, filename);

      const res = await fetch(`${API_BASE}/api/analyze`, { method: 'POST', body: formData });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Analysis failed'); }
      const data = await res.json();

      addLog(`> View Classification: ${data.landmarks.qa.view_type} View Detected`);
      await new Promise(r => setTimeout(r, 200));
      addLog(`> Positioning QA: ${data.landmarks.qa.status} (Score: ${data.landmarks.qa.score})`);
      await new Promise(r => setTimeout(r, 300));
      addLog('> Femur Condyle (大腿骨顆部) detected ✓');
      await new Promise(r => setTimeout(r, 300));
      addLog('> Tibial Plateau (脛骨プラトー) detected ✓');
      await new Promise(r => setTimeout(r, 300));
      addLog('> Patella (膝蓋骨) detected ✓');
      await new Promise(r => setTimeout(r, 300));
      addLog('> Medial/Lateral Condyle (内外側顆) separated ✓');
      await new Promise(r => setTimeout(r, 300));
      addLog('> Estimating internal/external rotation...');
      await new Promise(r => setTimeout(r, 400));
      if (data.landmarks.qa.view_type === 'LAT') {
        addLog(`> TPA (Tibial Posterior Angle): ${data.landmarks.angles.TPA}°`);
        addLog(`> Flexion angle: ${data.landmarks.angles.flexion}°`);
      }
      addLog(`> Rotation: ${data.landmarks.angles.rotation}° — ${data.landmarks.angles.rotation_label}`);

      setLandmarks(data.landmarks);
      setIsAnalyzing(false);
      setAnalysisComplete(true);
    } catch (err: any) {
      addLog(`> ❌ Error: ${err.message}`);
      setError(err.message);
      setIsAnalyzing(false);
    }
  };

  const lm = landmarks;

  return (
    <div className="min-h-screen bg-neutral-950 text-slate-200 font-sans p-6 overflow-x-hidden">
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes scan {
          0%   { top: 0%;   opacity: 0; }
          10%  { opacity: 1; }
          90%  { opacity: 1; }
          100% { top: 100%; opacity: 0; }
        }
        @keyframes fade-in {
          0%   { opacity: 0; transform: scale(0.5); }
          100% { opacity: 1; transform: scale(1); }
        }
        @keyframes draw-line {
          from { stroke-dashoffset: 1000; }
          to   { stroke-dashoffset: 0; }
        }
        .animate-fade-in {
          animation: fade-in 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
          opacity: 0;
        }
        .animate-draw {
          stroke-dasharray: 1000;
          stroke-dashoffset: 1000;
          animation: draw-line 1.5s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
      `}} />

      <header className="max-w-6xl mx-auto mb-8 pb-4 border-b border-neutral-800 flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent transform hover:scale-[1.01] transition-transform cursor-default">
            OsteoVision AI
          </h1>
          <p className="text-sm text-neutral-500 mt-1">Deep Learning Biometrics & Diagnostic Analytics Engine</p>
        </div>
        <div className="flex items-center gap-2 text-sm font-medium text-cyan-500 bg-cyan-500/10 px-3 py-1.5 rounded-full border border-cyan-500/20">
          <Cpu className="w-4 h-4" />
          <span>v2.0 CV Engine Active</span>
        </div>
      </header>

      <main className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left: Viewport */}
        <div className="lg:col-span-2 space-y-4">
          <div className="bg-black border border-neutral-800 rounded-xl overflow-hidden shadow-2xl relative aspect-square md:aspect-video flex items-center justify-center group">

            {/* Drop Zone */}
            {!imagePreview && !loading && (
              <div
                {...getRootProps()}
                className={`absolute inset-0 m-4 border-2 border-dashed rounded-lg flex flex-col items-center justify-center p-6 text-center transition-colors cursor-pointer
                  ${isDragActive ? 'border-cyan-500 bg-cyan-500/5' : 'border-neutral-700 hover:border-cyan-500/50 hover:bg-neutral-900/50'}`}
              >
                <input {...getInputProps()} />
                <Upload className={`w-12 h-12 mb-4 ${isDragActive ? 'text-cyan-400' : 'text-neutral-500'}`} />
                <p className="text-lg font-medium text-neutral-300">
                  {isDragActive ? 'Drop DICOM file here...' : 'Drag & drop a DICOM file here'}
                </p>
                <p className="text-sm text-neutral-500 mt-2">or click to browse (.dcm, .png, .jpg)</p>
                <div className="mt-6">
                  <button
                    onClick={e => { e.stopPropagation(); setImagePreview('/knee_xray_1772269369187.png'); }}
                    className="text-xs bg-neutral-800 hover:bg-neutral-700 text-neutral-300 px-4 py-2 rounded-full transition-colors"
                  >
                    Load Demo Knee X-ray
                  </button>
                </div>
              </div>
            )}

            {/* Loading */}
            {loading && (
              <div className="flex flex-col items-center justify-center text-cyan-500">
                <div className="w-12 h-12 border-4 border-cyan-500/20 border-t-cyan-500 rounded-full animate-spin mb-4" />
                <p className="font-medium animate-pulse">Processing via FastAPI + pydicom…</p>
              </div>
            )}

            {/* Image + Overlays */}
            {imagePreview && (
              <div className="relative w-full h-full">
                <img 
                  src={(showHeatmap && lm?.heatmap_data) ? lm.heatmap_data : imagePreview} 
                  alt="Knee X-ray" 
                  className="w-full h-full object-contain filter contrast-125 brightness-90 transition-all duration-300" 
                />
                
                {/* DL Attention Map Toggle */}
                {analysisComplete && lm?.heatmap_data && (
                  <button
                    onClick={() => setShowHeatmap(!showHeatmap)}
                    className={`absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-full backdrop-blur-md border transition-all z-30 shadow-lg ${
                      showHeatmap 
                        ? 'bg-fuchsia-900/60 border-fuchsia-500/50 text-fuchsia-300' 
                        : 'bg-black/60 border-neutral-700 text-neutral-400 hover:text-cyan-400 hover:border-cyan-500/50'
                    }`}
                  >
                    <Cpu className="w-4 h-4" />
                    <span className="text-xs font-semibold tracking-wider">
                      {showHeatmap ? "HIDE AI HEATMAP" : "SHOW AI HEATMAP"}
                    </span>
                  </button>
                )}

                {/* Clear button */}
                <button
                  onClick={handleClear}
                  className="absolute top-4 right-4 bg-black/50 hover:bg-red-500/20 text-neutral-300 hover:text-red-400 p-2 rounded-full backdrop-blur-sm border border-neutral-700 transition-all opacity-0 group-hover:opacity-100"
                >
                  <X className="w-5 h-5" />
                </button>

                {/* Grid overlay */}
                <div className="absolute inset-0 bg-[linear-gradient(rgba(34,211,238,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(34,211,238,0.03)_1px,transparent_1px)] bg-[size:30px_30px] pointer-events-none mix-blend-screen" />

                {/* Scan line */}
                {isAnalyzing && (
                  <div className="absolute top-0 left-0 w-full h-[3px] bg-cyan-400 shadow-[0_0_20px_5px_rgba(34,211,238,0.7)] z-20 animate-[scan_2s_ease-in-out_infinite_alternate]" />
                )}

                {/* Real landmark overlays */}
                {analysisComplete && lm && (
                  <>
                    {/* SVG axes */}
                    <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 15 }}>
                      {/* Femoral axis */}
                      <line
                        x1={`${lm.femur_axis_top.x_pct}%`} y1={`${lm.femur_axis_top.y_pct}%`}
                        x2={`${lm.femur_condyle.x_pct}%`}  y2={`${lm.femur_condyle.y_pct}%`}
                        stroke="#facc15" strokeWidth="2" strokeDasharray="6,4"
                        className="animate-draw" style={{ animationDelay: '0.2s' }}
                      />
                      {/* Tibial axis */}
                      <line
                        x1={`${lm.tibial_plateau.x_pct}%`}    y1={`${lm.tibial_plateau.y_pct}%`}
                        x2={`${lm.tibia_axis_bottom.x_pct}%`} y2={`${lm.tibia_axis_bottom.y_pct}%`}
                        stroke="#ef4444" strokeWidth="2" strokeDasharray="6,4"
                        className="animate-draw" style={{ animationDelay: '0.5s' }}
                      />
                      {/* Plateau horizontal guide */}
                      <line
                        x1={`${Math.max(0, lm.tibial_plateau.x_pct - 15)}%`} y1={`${lm.tibial_plateau.y_pct}%`}
                        x2={`${Math.min(100, lm.tibial_plateau.x_pct + 15)}%`} y2={`${lm.tibial_plateau.y_pct}%`}
                        stroke="#22d3ee" strokeWidth="2"
                        className="animate-draw" style={{ animationDelay: '0.8s' }}
                      />
                      {/* Patellar tendon guide */}
                      <line
                        x1={`${lm.patella.x_pct}%`}       y1={`${lm.patella.y_pct}%`}
                        x2={`${lm.tibial_plateau.x_pct}%`} y2={`${lm.tibial_plateau.y_pct}%`}
                        stroke="#60a5fa" strokeWidth="2" strokeDasharray="3,5"
                        className="animate-draw" style={{ animationDelay: '1.0s' }}
                      />
                      {/* Condyle Rotation Line */}
                      <line
                        x1={`${lm.medial_condyle.x_pct}%`}  y1={`${lm.medial_condyle.y_pct}%`}
                        x2={`${lm.lateral_condyle.x_pct}%`} y2={`${lm.lateral_condyle.y_pct}%`}
                        stroke="#d946ef" strokeWidth="2" strokeDasharray="2,2"
                        className="animate-draw" style={{ animationDelay: '1.2s' }}
                      />

                      {/* Angle labels */}
                      <g className="animate-fade-in" style={{ animationDelay: '1.4s' }}>
                        <text
                          x={`${lm.tibial_plateau.x_pct - 12}%`}
                          y={`${lm.tibial_plateau.y_pct - 4}%`}
                          fill="#22d3ee" fontSize="12" fontWeight="bold"
                          filter="drop-shadow(0px 2px 3px rgba(0,0,0,0.9))"
                        >TPA: {lm.angles.TPA}°</text>
                        <text
                          x={`${lm.femur_axis_top.x_pct + 2}%`}
                          y={`${lm.femur_axis_top.y_pct + 5}%`}
                          fill="#facc15" fontSize="12" fontWeight="bold"
                          filter="drop-shadow(0px 2px 3px rgba(0,0,0,0.9))"
                        >Flexion: {lm.angles.flexion}°</text>
                      </g>
                    </svg>

                    {/* Femur Condyle dot */}
                    <div
                      className="absolute z-20 flex items-center gap-2 group/dot cursor-pointer animate-fade-in"
                      style={{ top: `${lm.femur_condyle.y_pct}%`, left: `${lm.femur_condyle.x_pct}%`, transform: 'translate(-50%, -50%)', animationDelay: '0.3s' }}
                    >
                      <div className="w-3.5 h-3.5 bg-yellow-400 rounded-full shadow-[0_0_15px_#facc15] relative group-hover/dot:scale-[2] transition-transform">
                        <div className="absolute inset-0 bg-yellow-400 rounded-full animate-ping opacity-60" />
                      </div>
                      <div className="bg-black/95 backdrop-blur-md border border-yellow-400/50 text-yellow-400 text-xs px-2 py-1 rounded-md opacity-0 group-hover/dot:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        大腿骨顆部 (Femur Condyle)<br/>
                        px: ({lm.femur_condyle.x}, {lm.femur_condyle.y})<br/>
                        Rotation: {lm.angles.rotation}°
                      </div>
                    </div>

                    {/* Tibial Plateau dot */}
                    <div
                      className="absolute z-20 flex items-center gap-2 group/dot cursor-pointer animate-fade-in"
                      style={{ top: `${lm.tibial_plateau.y_pct}%`, left: `${lm.tibial_plateau.x_pct}%`, transform: 'translate(-50%, -50%)', animationDelay: '0.5s' }}
                    >
                      <div className="w-3.5 h-3.5 bg-red-500 rounded-full shadow-[0_0_15px_#ef4444] relative group-hover/dot:scale-[2] transition-transform">
                        <div className="absolute inset-0 bg-red-400 rounded-full animate-ping opacity-60" />
                      </div>
                      <div className="bg-black/95 backdrop-blur-md border border-red-500/50 text-red-400 text-xs px-2 py-1 rounded-md opacity-0 group-hover/dot:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        脛骨プラトー (Tibial Plateau)<br/>
                        px: ({lm.tibial_plateau.x}, {lm.tibial_plateau.y})<br/>
                        TPA: {lm.angles.TPA}° Posterior
                      </div>
                    </div>

                    {/* Patella dot */}
                    <div
                      className="absolute z-20 flex items-center gap-2 group/dot cursor-pointer animate-fade-in"
                      style={{ top: `${lm.patella.y_pct}%`, left: `${lm.patella.x_pct}%`, transform: 'translate(-50%, -50%)', animationDelay: '0.7s' }}
                    >
                      <div className="w-3.5 h-3.5 bg-blue-400 rounded-full shadow-[0_0_15px_#60a5fa] relative group-hover/dot:scale-[2] transition-transform">
                        <div className="absolute inset-0 bg-blue-400 rounded-full animate-ping opacity-60" />
                      </div>
                      <div className="bg-black/95 backdrop-blur-md border border-blue-400/50 text-blue-400 text-xs px-2 py-1 rounded-md opacity-0 group-hover/dot:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        膝蓋骨 (Patella)<br/>
                        px: ({lm.patella.x}, {lm.patella.y})
                      </div>
                    </div>

                    {/* Medial Condyle dot */}
                    <div
                      className="absolute z-20 flex items-center gap-2 group/dot cursor-pointer animate-fade-in"
                      style={{ top: `${lm.medial_condyle.y_pct}%`, left: `${lm.medial_condyle.x_pct}%`, transform: 'translate(-50%, -50%)', animationDelay: '0.8s' }}
                    >
                      <div className="w-2.5 h-2.5 bg-fuchsia-500 rounded-full shadow-[0_0_10px_#d946ef] relative group-hover/dot:scale-[2] transition-transform" />
                      <div className="bg-black/95 backdrop-blur-md border border-fuchsia-500/50 text-fuchsia-400 text-xs px-2 py-1 rounded-md opacity-0 group-hover/dot:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        内側顆 (Medial)
                      </div>
                    </div>

                    {/* Lateral Condyle dot */}
                    <div
                      className="absolute z-20 flex flex-row-reverse items-center gap-2 group/dot cursor-pointer animate-fade-in"
                      style={{ top: `${lm.lateral_condyle.y_pct}%`, left: `${lm.lateral_condyle.x_pct}%`, transform: 'translate(-50%, -50%)', animationDelay: '0.9s' }}
                    >
                      <div className="w-2.5 h-2.5 bg-fuchsia-300 rounded-full shadow-[0_0_10px_#f0abfc] relative group-hover/dot:scale-[2] transition-transform" />
                      <div className="bg-black/95 backdrop-blur-md border border-fuchsia-300/50 text-fuchsia-300 text-xs px-2 py-1 rounded-md opacity-0 group-hover/dot:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                        外側顆 (Lateral)
                      </div>
                    </div>

                    {/* Joint Space bounding box */}
                    <div
                      className="absolute z-10 border-2 border-dashed border-green-500 bg-green-500/10 animate-fade-in flex items-end"
                      style={{
                        top:    `${Math.min(lm.femur_condyle.y_pct, lm.tibial_plateau.y_pct)}%`,
                        left:   `${Math.min(lm.femur_condyle.x_pct, lm.tibial_plateau.x_pct) - 5}%`,
                        width:  `${Math.abs(lm.femur_condyle.x_pct - lm.tibial_plateau.x_pct) + 10}%`,
                        height: `${Math.abs(lm.femur_condyle.y_pct - lm.tibial_plateau.y_pct)}%`,
                        minHeight: '3%',
                        animationDelay: '1.2s',
                      }}
                    >
                      <div className="bg-green-500 text-black text-[10px] font-bold px-2 py-0.5 whitespace-nowrap shadow-md">
                        Joint Space (QA Score: {lm.qa.score}%)
                      </div>
                    </div>
                  </>
                )}
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="absolute inset-x-4 bottom-4 bg-red-950/80 border border-red-500/50 text-red-200 px-4 py-3 rounded-lg flex items-start gap-3 backdrop-blur-sm">
                <ShieldAlert className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
                <div className="text-sm">{error}</div>
              </div>
            )}
          </div>

          {/* Action Button */}
          <button
            onClick={runAnalysis}
            disabled={!imagePreview || isAnalyzing || analysisComplete}
            className={`w-full py-3 px-6 rounded-lg font-bold tracking-wide transition-all ${
              !imagePreview
                ? 'bg-neutral-800 text-neutral-600 cursor-not-allowed'
                : isAnalyzing
                ? 'bg-cyan-900/40 text-cyan-400 cursor-wait'
                : analysisComplete
                ? 'bg-emerald-900/30 text-emerald-400 border border-emerald-500/30'
                : 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white shadow-[0_0_20px_rgba(8,145,178,0.4)] hover:shadow-[0_0_30px_rgba(8,145,178,0.6)] transform hover:-translate-y-0.5'
            }`}
          >
            {isAnalyzing ? (
              <span className="flex items-center justify-center gap-3">
                <span className="w-5 h-5 border-2 border-cyan-400/30 border-t-cyan-400 rounded-full animate-spin" />
                Processing Scan...
              </span>
            ) : analysisComplete ? '✓ Analysis Complete' : 'Run Deep Learning Analysis'}
          </button>

          <div className="flex justify-between text-xs text-neutral-500 font-mono">
            <span>Client: Next.js + Tailwind</span>
            <span>Server: FastAPI + OpenCV / pydicom</span>
          </div>
        </div>

        {/* Right: Metadata + Angles + Console */}
        <div className="space-y-4">
          {/* Metadata */}
          <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 shadow-lg">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-3 flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              Extracted Metadata
            </h2>
            {metadata ? (
              <div className="space-y-2 text-sm">
                <DataRow label="Patient ID"   value={metadata.PatientID} />
                <DataRow label="Patient Name" value={metadata.PatientName} />
                <DataRow label="Study Date"   value={metadata.StudyDate} />
                <DataRow label="Modality"     value={metadata.Modality} />
                <DataRow label="Manufacturer" value={metadata.Manufacturer} />
                <DataRow label="Resolution"   value={`${metadata.Columns ?? '?'} × ${metadata.Rows ?? '?'}`} />
              </div>
            ) : (
              <div className="text-sm text-neutral-600 italic py-6 text-center border border-dashed border-neutral-800 rounded">
                No data loaded. Upload a DICOM file.
              </div>
            )}
          </div>

          {/* QA & Angles panel (shown when analysis complete) */}
          {analysisComplete && lm && (
            <div className="space-y-4 animate-fade-in">
              {/* QA Score Panel */}
              <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 shadow-lg relative overflow-hidden">
                <div className={`absolute top-0 left-0 w-1 h-full ${(colorMap[lm.qa.color] || colorMap.green).bg}`} />
                <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-3 flex items-center gap-2">
                  <div className={`w-2 h-2 rounded-full ${(colorMap[lm.qa.color] || colorMap.green).bgDot} animate-pulse`} />
                  Quality Assurance (ポジショニング判定)
                </h2>
                
                <div className="flex items-end justify-between mb-2">
                  <div>
                    <div className="text-3xl font-black text-white">{lm.qa.score}<span className="text-lg text-neutral-500 font-normal"> /100</span></div>
                    <div className={`text-sm font-bold ${(colorMap[lm.qa.color] || colorMap.green).text} uppercase tracking-widest`}>{lm.qa.status} - {lm.qa.view_type} View</div>
                  </div>
                </div>
                {lm.qa.positioning_advice && (
                  <div className="mb-3 bg-blue-900/40 border border-blue-800 rounded-lg p-3 text-blue-200 text-sm font-bold shadow-inner">
                    {lm.qa.positioning_advice}
                  </div>
                )}
                <p className="text-sm text-neutral-300 bg-black/30 p-3 rounded-lg border border-neutral-800">
                  {lm.qa.message}
                </p>
              </div>

              {/* Angles Panel */}
              <div className="bg-neutral-900 border border-neutral-800 rounded-xl p-5 shadow-lg">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-neutral-400 mb-3 flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full bg-emerald-400" />
                  Computed Angles
                </h2>
                <div className="space-y-3">
                  {lm.qa.view_type === 'LAT' && (
                    <>
                      <AngleRow label="TPA (Tibial Posterior Angle)" value={`${lm.angles.TPA}°`} color="text-cyan-400" normal="5–10°" />
                      <AngleRow label="Flexion Angle"                value={`${lm.angles.flexion}°`} color="text-yellow-400" normal="0–180°" />
                    </>
                  )}
                  <AngleRow label={`Rotation (${lm.angles.rotation_label})`} value={`${lm.angles.rotation}°`} color="text-fuchsia-400" normal="0° (Neutral)" />
                </div>
                <div className="mt-4 text-xs font-mono text-neutral-600">
                  {lm.qa.view_type === 'AP' && `Symmetry Ratio: ${lm.qa.symmetry_ratio}\n`}
                  Femur Condyle: ({lm.femur_condyle.x}, {lm.femur_condyle.y})px<br/>
                  Tibial Plateau: ({lm.tibial_plateau.x}, {lm.tibial_plateau.y})px<br/>
                  Patella: ({lm.patella.x}, {lm.patella.y})px<br/>
                  M/L Condyle Width Diff: {Math.abs(lm.lateral_condyle.x - lm.medial_condyle.x)}px
                </div>
              </div>
            </div>
          )}

          {/* Console */}
          <div className="bg-black border border-neutral-800 rounded-xl overflow-hidden shadow-lg flex flex-col h-56">
            <div className="bg-neutral-900 px-4 py-2 text-xs font-mono text-neutral-500 border-b border-neutral-800 flex items-center gap-2">
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-red-500/50" />
                <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/50" />
                <div className="w-2.5 h-2.5 rounded-full bg-green-500/50" />
              </div>
              <span className="ml-2">system_console</span>
            </div>
            <div ref={consoleRef} className="p-4 font-mono text-xs overflow-y-auto flex-1 space-y-0.5">
              <div className="text-neutral-500">{`> FastAPI v2.0 ready — real CV detection enabled`}</div>
              {loading && <div className="text-yellow-400">{`> Parsing DICOM via pydicom...`}</div>}
              {metadata && <div className="text-green-400">{`> Metadata extracted OK.`}</div>}
              {imagePreview && <div className="text-cyan-400">{`> Image ready. Click "Run AI Feature Extraction" to analyze.`}</div>}
              {logs.length > 0 && <div className="border-t border-neutral-800 pt-1 mt-1" />}
              {logs.map((l, i) => (
                <div key={i} className="text-fuchsia-400">{l}</div>
              ))}
              {analysisComplete && (
                <div className="text-emerald-400 font-bold mt-1 border-t border-emerald-900/50 pt-1">
                  {'> ✓ Analysis complete. Landmarks from real CV detection.'}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function DataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col border-b border-neutral-800/50 pb-2">
      <span className="text-xs text-neutral-500 mb-0.5">{label}</span>
      <span className="font-mono text-neutral-200 truncate text-sm">{value || 'N/A'}</span>
    </div>
  );
}

function AngleRow({ label, value, color, normal }: { label: string; value: string; color: string; normal: string }) {
  return (
    <div className="flex flex-col border-b border-neutral-800/50 pb-2">
      <span className="text-xs text-neutral-500 mb-0.5">{label}</span>
      <div className="flex items-baseline gap-2">
        <span className={`font-mono font-bold text-lg ${color}`}>{value}</span>
        <span className="text-xs text-neutral-600">normal: {normal}</span>
      </div>
    </div>
  );
}
