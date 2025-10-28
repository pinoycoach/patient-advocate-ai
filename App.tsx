
import React, { useState, useEffect, useCallback, useRef } from 'react';
import { callGemini, callGeminiProofread, GeminiResponseData } from './services/geminiService';
import Spinner from './components/Spinner';
import ResultSection from './components/ResultSection';
import Disclaimer from './components/Disclaimer';
import ErrorMessage from './components/ErrorMessage';
import { GoogleGenAI, LiveServerMessage, Modality } from "@google/genai";

type TabName = 'translate' | 'prepare' | 'summarize' | 'labs' | 'record';

// Declare window.aistudio for TypeScript
declare global {
  interface AIStudio {
    hasSelectedApiKey: () => Promise<boolean>;
    openSelectKey: () => Promise<void>;
  }

  interface Window {
    aistudio?: AIStudio;
    // Fix: Declare webkitAudioContext to resolve TypeScript error, maintaining cross-browser compatibility as per guidelines.
    webkitAudioContext?: typeof AudioContext; 
  }
}

// Utility functions for audio decoding/encoding as per @google/genai Live API guidelines
function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function createBlob(data: Float32Array): { data: string; mimeType: string } {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// Helper for inline markdown (bold/italic)
const parseInlineMarkdown = (text: string): React.ReactNode => {
  let html = text;

  // Handle bold first to prevent it from being caught by italic regex
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Handle italic (single asterisks) - ensuring it's not part of a bold tag
  html = html.replace(/(?<!\*)\*([^*]+?)\*(?!\*)/g, '<em>$1</em>');
  
  // Handle italic (underscores)
  html = html.replace(/\_([^_]+?)\_/g, '<em>$1</em>');

  return <span dangerouslySetInnerHTML={{ __html: html }} />;
};


// Helper to convert simple markdown to React nodes
const renderMarkdown = (markdownText: string): React.ReactNode => {
  if (!markdownText) return null;

  const lines = markdownText.split('\n');
  const elements: React.ReactNode[] = [];
  let currentList: { type: 'ol' | 'ul', items: React.ReactNode[] } | null = null;

  const closeCurrentList = () => {
    if (currentList) {
      if (currentList.type === 'ol') {
        elements.push(<ol key={`ol-block-${elements.length}`} className="list-decimal pl-5 mb-2">{currentList.items}</ol>);
      } else {
        elements.push(<ul key={`ul-block-${elements.length}`} className="list-disc pl-5 mb-2">{currentList.items}</ul>);
      }
      currentList = null;
    }
  };

  lines.forEach((line, index) => {
    const trimmedLine = line.trim();

    // Headers (e.g., ## My Header) - We'll use h4 here to avoid conflicting with ResultSection titles
    const headerMatch = trimmedLine.match(/^(#+)\s*(.*)/);
    if (headerMatch) {
      closeCurrentList();
      const level = Math.min(headerMatch[1].length, 3); // Max h3 within a ResultSection
      const headerText = parseInlineMarkdown(headerMatch[2]);
      if (level === 1) elements.push(<h3 key={`h3-${index}`} className="text-xl font-semibold mb-2 mt-4 text-primary">{headerText}</h3>);
      else if (level === 2) elements.push(<h4 key={`h4-${index}`} className="text-lg font-semibold mb-2 mt-3 text-primary">{headerText}</h4>);
      else elements.push(<h5 key={`h5-${index}`} className="text-base font-semibold mb-2 mt-2">{headerText}</h5>);
      return;
    }

    // Ordered list items
    const olMatch = trimmedLine.match(/^(\d+)\.\s*(.*)/);
    if (olMatch) {
      if (!currentList || currentList.type !== 'ol') {
        closeCurrentList();
        currentList = { type: 'ol', items: [] };
      }
      currentList.items.push(<li key={`li-${index}`}>{parseInlineMarkdown(olMatch[2])}</li>);
      return;
    }

    // Unordered list items
    const ulMatch = trimmedLine.match(/^[-*]\s*(.*)/);
    if (ulMatch) {
      if (!currentList || currentList.type !== 'ul') {
        closeCurrentList();
        currentList = { type: 'ul', items: [] };
      }
      currentList.items.push(<li key={`li-${index}`}>{parseInlineMarkdown(ulMatch[1])}</li>);
      return;
    }

    // Paragraphs or empty lines
    closeCurrentList();
    if (trimmedLine) {
      elements.push(<p key={`p-${index}`} className="mb-2">{parseInlineMarkdown(trimmedLine)}</p>);
    } else {
      elements.push(<br key={`br-${index}`} />);
    }
  });

  closeCurrentList(); // Close any remaining open list

  return <>{elements}</>;
};


const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabName>('translate');
  const [apiConfigured, setApiConfigured] = useState<boolean>(false);
  const [apiKeyStatusMessage, setApiKeyStatusMessage] = useState<string>('');
  const [showSelectKeyButton, setShowSelectKeyButton] = useState<boolean>(false);
  const [apiIsSelecting, setApiIsSelecting] = useState<boolean>(false);

  // States for each feature
  const [translateInput, setTranslateInput] = useState<string>('');
  const [translateOutput, setTranslateOutput] = useState<string | null>(null);
  const [translateLoading, setTranslateLoading] = useState<boolean>(false);
  const [translateError, setTranslateError] = useState<string | null>(null);
  // States for Translate Jargon Proofreading
  const [translateProofreadingResult, setTranslateProofreadingResult] = useState<string | null>(null);
  const [translateProofreadingLoading, setTranslateProofreadingLoading] = useState<boolean>(false);
  const [translateProofreadingError, setTranslateProofreadingError] = useState<string | null>(null);


  const [prepareInput, setPrepareInput] = useState<string>('');
  const [prepareOutput, setPrepareOutput] = useState<string | null>(null);
  const [prepareLoading, setPrepareLoading] = useState<boolean>(false);
  const [prepareError, setPrepareError] = useState<string | null>(null);
  const [prepareGroundingUrls, setPrepareGroundingUrls] = useState<{ uri: string; title?: string }[] | null>(null);
  // States for Prepare Appointment Proofreading
  const [prepareProofreadingResult, setPrepareProofreadingResult] = useState<string | null>(null);
  const [prepareProofreadingLoading, setPrepareProofreadingLoading] = useState<boolean>(false);
  const [prepareProofreadingError, setPrepareProofreadingError] = useState<string | null>(null);


  const [summarizeInput, setSummarizeInput] = useState<string>('');
  const [summarizeOutput, setSummarizeOutput] = useState<string | null>(null);
  const [summarizeLoading, setSummarizeLoading] = useState<boolean>(false);
  const [summarizeError, setSummarizeError] = useState<string | null>(null);
  const [summarizeImageFile, setSummarizeImageFile] = useState<File | null>(null); // State for image file
  const [summarizeImagePreviewUrl, setSummarizeImagePreviewUrl] = useState<string | null>(null); // State for image preview
  // States for Summarize Notes Proofreading
  const [summarizeProofreadingResult, setSummarizeProofreadingResult] = useState<string | null>(null);
  const [summarizeProofreadingLoading, setSummarizeProofreadingLoading] = useState<boolean>(false);
  const [summarizeProofreadingError, setSummarizeProofreadingError] = useState<string | null>(null);
  const [extractedMedications, setExtractedMedications] = useState<string | null>(null); // New state for medications


  const [labsInput, setLabsInput] = useState<string>('');
  const [labsOutput, setLabsOutput] = useState<string | null>(null);
  const [labsLoading, setLabsLoading] = useState<boolean>(false);
  const [labsError, setLabsError] = useState<string | null>(null);
  const [labsGroundingUrls, setLabsGroundingUrls] = useState<{ uri: string; title?: string }[] | null>(null);
  const [labsImageFile, setLabsImageFile] = useState<File | null>(null); // New state for image file
  const [labsImagePreviewUrl, setLabsImagePreviewUrl] = useState<string | null>(null); // New state for image preview
  // States for Explain Labs Proofreading
  const [labsProofreadingResult, setLabsProofreadingResult] = useState<string | null>(null);
  const [labsProofreadingLoading, setLabsProofreadingLoading] = useState<boolean>(false);
  const [labsProofreadingError, setLabsProofreadingError] = useState<string | null>(null);


  // States for Record Session feature
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [transcriptionHistory, setTranscriptionHistory] = useState<{ speaker: string; text: string }[]>([]);
  const [currentLiveInputTranscription, setCurrentLiveInputTranscription] = useState<string>('');
  const [currentLiveOutputTranscription, setCurrentLiveOutputTranscription] = useState<string>('');

  // Fix: New refs to hold the latest transcription state to avoid stale closures in `onmessage`
  const latestInputTranscriptionRef = useRef('');
  const latestOutputTranscriptionRef = useRef('');

  // Fix: Sync current transcription states with refs
  useEffect(() => {
    latestInputTranscriptionRef.current = currentLiveInputTranscription;
  }, [currentLiveInputTranscription]);

  useEffect(() => {
    latestOutputTranscriptionRef.current = currentLiveOutputTranscription;
  }, [currentLiveOutputTranscription]);


  const [liveLoading, setLiveLoading] = useState<boolean>(false);
  const [liveError, setLiveError] = useState<string | null>(null);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const outputNodeRef = useRef<GainNode | null>(null); // Added for consistency with guidelines
  const scriptProcessorNodeRef = useRef<ScriptProcessorNode | null>(null);
  const liveSessionPromiseRef = useRef<Promise<any> | null>(null);
  const liveSessionRef = useRef<any | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const audioSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  // Fix: Moved handleStopRecording before its usages in other useCallback hooks.
  const handleStopRecording = useCallback(() => {
    if (liveSessionRef.current) {
      liveSessionRef.current.close();
      liveSessionRef.current = null;
      liveSessionPromiseRef.current = null; // Clear promise reference
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    if (scriptProcessorNodeRef.current) {
      scriptProcessorNodeRef.current.disconnect();
      scriptProcessorNodeRef.current.onaudioprocess = null;
      scriptProcessorNodeRef.current = null;
    }

    if (inputAudioContextRef.current) {
      inputAudioContextRef.current.close();
      inputAudioContextRef.current = null;
    }
    if (outputAudioContextRef.current) {
      outputAudioContextRef.current.close();
      outputAudioContextRef.current = null;
    }
    if (outputNodeRef.current) {
      outputNodeRef.current.disconnect();
      outputNodeRef.current = null;
    }

    // Stop all playing audio sources
    for (const source of audioSourcesRef.current.values()) {
      source.stop();
    }
    audioSourcesRef.current.clear();
    nextStartTimeRef.current = 0;

    setIsRecording(false);
    setLiveLoading(false);
    // Finalize any ongoing transcriptions using the latest ref values
    // Fix: Use latest ref values for final history update and clear them
    if (latestInputTranscriptionRef.current) {
      setTranscriptionHistory(prev => [...prev, { speaker: 'User', text: latestInputTranscriptionRef.current.trim() }]);
      setCurrentLiveInputTranscription('');
      latestInputTranscriptionRef.current = ''; // Clear ref too
    }
    if (latestOutputTranscriptionRef.current) {
      setTranscriptionHistory(prev => [...prev, { speaker: 'AI', text: latestOutputTranscriptionRef.current.trim() }]);
      setCurrentLiveOutputTranscription('');
      latestOutputTranscriptionRef.current = ''; // Clear ref too
    }

  }, [latestInputTranscriptionRef, latestOutputTranscriptionRef, setTranscriptionHistory, setCurrentLiveInputTranscription, setCurrentLiveOutputTranscription, setIsRecording, setLiveLoading]); // Added setters to dependencies as they are called directly within this callback.

  const checkApiKeyStatus = useCallback(async () => {
    setApiConfigured(false);
    setShowSelectKeyButton(false);
    setApiKeyStatusMessage('');

    if (window.aistudio && typeof window.aistudio.hasSelectedApiKey === 'function') {
      try {
        const hasSelected = await window.aistudio.hasSelectedApiKey();
        if (hasSelected) {
          setApiConfigured(true);
          setApiKeyStatusMessage('✅ Gemini AI Ready - All features enabled');
        } else {
          setApiConfigured(false);
          setApiKeyStatusMessage(
            `🔑 Gemini API Key not configured. Please select your API key to get started.`
          );
          setShowSelectKeyButton(true);
        }
      } catch (e: any) {
        console.error("Error checking aistudio API key status:", e);
        if (process.env.API_KEY && process.env.API_KEY.startsWith('AIza')) {
          setApiConfigured(true);
          setApiKeyStatusMessage('✅ Gemini AI Ready - All features enabled (aistudio check failed)');
        } else {
          setApiConfigured(false);
          setApiKeyStatusMessage(
            `Gemini API key is not configured. Please set the API_KEY environment variable. 
            For local development, you might need to configure it in your build tool (e.g., VITE_API_KEY for Vite apps).`
          );
        }
      }
    } else {
      if (process.env.API_KEY && process.env.API_KEY.startsWith('AIza')) {
        setApiConfigured(true);
        setApiKeyStatusMessage('✅ Gemini AI Ready - All features enabled');
      } else {
        setApiConfigured(false);
        setApiKeyStatusMessage(
          `Gemini API key is not configured. Please set the API_KEY environment variable. 
          For local development, you might need to configure it in your build tool (e.g., VITE_API_KEY for Vite apps).`
        );
      }
    }
  }, []); // Setters are stable, no need to list.

  useEffect(() => {
    checkApiKeyStatus();
  }, [checkApiKeyStatus]);

  const handleSelectApiKey = useCallback(async () => {
    if (window.aistudio && typeof window.aistudio.openSelectKey === 'function' && !apiIsSelecting) {
      setApiIsSelecting(true);
      try {
        await window.aistudio.openSelectKey();
        await checkApiKeyStatus(); // Await the status check to ensure UI updates
      } catch (e: any) {
        console.error("Error opening API key selection dialog:", e);
        setApiKeyStatusMessage(`Failed to open API key selection: ${e.message}`);
      } finally {
        setApiIsSelecting(false);
      }
    }
  }, [apiIsSelecting, checkApiKeyStatus, setApiIsSelecting, setApiKeyStatusMessage]); // Added setters to dependencies.

  const handleTabChange = useCallback((tab: TabName) => {
    setActiveTab(tab);
    // When switching tabs, clear recording if active
    if (isRecording) {
      handleStopRecording();
    }
  }, [isRecording, handleStopRecording, setActiveTab]); // Added setters to dependencies.

  // Fix: Use a type alias for the generic function signature to avoid potential parsing issues with generics in `useCallback`'s immediate function.
  type ExecuteGeminiCallFunc = <T>(action: () => Promise<T>, setError: (msg: string | null) => void) => Promise<T | undefined>;

  const executeGeminiCall: ExecuteGeminiCallFunc = useCallback(
    async (action, setError) => { // T is inferred from 'action'
      setError(null); // Clear previous error at the start of execution
      try {
        const result = await action();
        return result;
      } catch (error: any) {
        let errorMessageText = error instanceof Error ? error.message : String(error);

        // Check for API key configuration error coming from geminiService.ts
        if (errorMessageText.includes("Gemini API key is not configured.") && window.aistudio) {
          errorMessageText = "Gemini API key is not configured. Please select your API key again from the platform dialog.";
          console.warn("API key not configured internally, prompting re-selection via aistudio.");
          setError(errorMessageText);
          await checkApiKeyStatus(); // Trigger re-check to show select key button
        }
        // Check for "Requested entity was not found." which often indicates an invalid/expired API key in the API response itself
        else if (errorMessageText.includes("Requested entity was not found.") && window.aistudio) {
          errorMessageText = "Your API key might be invalid or expired. Please re-select it.";
          console.warn("API returned 'Requested entity not found', prompting re-selection.");
          setError(errorMessageText);
          await checkApiKeyStatus(); // Trigger re-check to show select key button
        } else {
          setError(errorMessageText);
        }
        // Return undefined on error to indicate failure to the caller
        return undefined; 
      }
    },
    [checkApiKeyStatus] // checkApiKeyStatus is a dependency. setError is a setter, so stable.
  );


  const handleTranslateJargon = useCallback(async () => {
    if (!translateInput.trim()) {
      alert('Please enter some medical text to translate');
      return;
    }
    setTranslateLoading(true);
    setTranslateOutput(null);
    setTranslateError(null);
    setTranslateProofreadingResult(null); // Clear proofread result on new submission


    try {
      const prompt = `You are a patient advocate helping people understand medical information.

Medical text: "${translateInput}"

Please provide:
1. A simple, plain-language explanation of what this means (2-3 sentences)
2. 5 important questions the patient should ask their doctor about this

Format your response as:
EXPLANATION:
[Your simple explanation here]

QUESTIONS TO ASK:
1. [Question 1]
2. [Question 2]
3. [Question 3]
4. [Question 4]
5. [Question 5]`;

      const result = await executeGeminiCall(() => callGemini({ prompt }), setTranslateError);
      if (result) {
        setTranslateOutput(result.text);
      }
    } catch (error) {
      // Error already handled by executeGeminiCall
    } finally {
      setTranslateLoading(false);
    }
  }, [translateInput, executeGeminiCall, setTranslateLoading, setTranslateOutput, setTranslateError, setTranslateProofreadingResult]); // Added all state variables and setters to dependencies

  const handlePrepareAppointment = useCallback(async () => {
    if (!prepareInput.trim()) {
      alert('Please describe your symptoms');
      return;
    }
    setPrepareLoading(true);
    setPrepareOutput(null);
    setPrepareError(null);
    setPrepareGroundingUrls(null); // Clear previous URLs
    setPrepareProofreadingResult(null); // Clear proofread result on new submission


    try {
      const prompt = `You are a patient advocate helping someone prepare for a doctor's appointment. Provide up-to-date and accurate information.

Symptoms: "${prepareInput}"

Create a comprehensive appointment preparation guide with:
1. Key information to tell the doctor (5-6 specific points)
2. Important questions to ask (5-7 questions)
3. Red flags to emphasize (2-3 points)

Format clearly with headers and bullet points using markdown.`; // Added markdown instruction

      const result = await executeGeminiCall(
        () => callGemini({ prompt, useSearchGrounding: true }), // Enable search grounding
        setPrepareError
      );
      if (result) {
        setPrepareOutput(result.text);
        setPrepareGroundingUrls(result.groundingUrls || null); // Store URLs
      }
    } catch (error) {
      // Error already handled by executeGeminiCall
    } finally {
      setPrepareLoading(false);
    }
  }, [prepareInput, executeGeminiCall, setPrepareLoading, setPrepareOutput, setPrepareError, setPrepareGroundingUrls, setPrepareProofreadingResult]); // Added all state variables and setters to dependencies

  // Reusable image handling logic
  const handleImageChange = useCallback((event: React.ChangeEvent<HTMLInputElement>, setImageFile: React.Dispatch<React.SetStateAction<File | null>>, setImagePreviewUrl: React.Dispatch<React.SetStateAction<string | null>>) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setImageFile(file);
      setImagePreviewUrl(URL.createObjectURL(file));
    } else {
      setImageFile(null);
      setImagePreviewUrl(null);
    }
  }, []); // Setters are passed as arguments.

  const handleClearImage = useCallback((setImageFile: React.Dispatch<React.SetStateAction<File | null>>, setImagePreviewUrl: React.Dispatch<React.SetStateAction<string | null>>, inputElementId: string) => {
    setImageFile(null);
    setImagePreviewUrl(null);
    // Also clear the file input value
    const fileInput = document.getElementById(inputElementId) as HTMLInputElement;
    if (fileInput) fileInput.value = '';
  }, []); // Setters are passed as arguments.

  // Specific image handlers for Summarize
  const handleSummarizeImageChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    handleImageChange(event, setSummarizeImageFile, setSummarizeImagePreviewUrl);
  }, [handleImageChange, setSummarizeImageFile, setSummarizeImagePreviewUrl]); // Added setters to dependencies.

  const handleClearSummarizeImage = useCallback(() => {
    handleClearImage(setSummarizeImageFile, setSummarizeImagePreviewUrl, 'summarize-image-input');
  }, [handleClearImage, setSummarizeImageFile, setSummarizeImagePreviewUrl]); // Added setters to dependencies.

  // Specific image handlers for Labs
  const handleLabsImageChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    handleImageChange(event, setLabsImageFile, setLabsImagePreviewUrl);
  }, [handleImageChange, setLabsImageFile, setLabsImagePreviewUrl]); // Added setters to dependencies.

  const handleClearLabsImage = useCallback(() => {
    handleClearImage(setLabsImageFile, setLabsImagePreviewUrl, 'labs-image-input');
  }, [handleClearImage, setLabsImageFile, setLabsImagePreviewUrl]); // Added setters to dependencies.


  const handleSummarizeNotes = useCallback(async () => {
    if (!summarizeInput.trim() && !summarizeImageFile) {
      alert('Please paste your appointment notes or upload an image of them');
      return;
    }
    setSummarizeLoading(true);
    setSummarizeOutput(null);
    setSummarizeError(null);
    setSummarizeProofreadingResult(null); // Clear proofread result on new submission
    setExtractedMedications(null); // Clear previous medication extraction


    try {
      const prompt = `You are a patient advocate helping someone understand their medical appointment notes or prescription.
Analyze the provided information (text and/or image). If no clear medical information is found in the input, please state "No clear medical information found in the provided input."

If it's a prescription:
- Identify medication name(s), dosage, frequency, and any specific instructions.
- Provide a clear, simple explanation of the medication's purpose (general, not patient-specific).

If it's general notes:
- Provide a clear summary of what the doctor said (2-3 sentences).
- List specific action items (medications, tests, follow-ups).
- Indicate when to follow up or call the doctor.

Combine these into a patient-friendly response. Use markdown for lists and emphasis.
Specifically, if medications are identified, format them under a "MEDICATION DETAILS:" header.

Doctor's notes/prescription (text): "${summarizeInput}"
(If an image is also provided, consider it part of the notes/prescription.)`;

      const result = await executeGeminiCall(
        () => callGemini({ prompt, imageFile: summarizeImageFile || undefined }),
        setSummarizeError
      );
      
      if (result) {
        setSummarizeOutput(result.text);

        // Attempt to extract medication details for dedicated display
        const medicationRegex = /MEDICATION DETAILS:([\s\S]*)/i;
        const medicationMatch = result.text.match(medicationRegex);
        if (medicationMatch && medicationMatch[1].trim()) {
          setExtractedMedications(medicationMatch[1].trim());
        }
      }
    } catch (error) {
      // Error already handled by executeGeminiCall
    } finally {
      setSummarizeLoading(false);
    }
  }, [summarizeInput, summarizeImageFile, executeGeminiCall, setSummarizeLoading, setSummarizeOutput, setSummarizeError, setSummarizeProofreadingResult, setExtractedMedications]); // Added all state variables and setters to dependencies

  const handleExplainLabs = useCallback(async () => {
    if (!labsInput.trim() && !labsImageFile) {
      alert('Please paste your lab results or upload an image of them');
      return;
    }
    setLabsLoading(true);
    setLabsOutput(null);
    setLabsError(null);
    setLabsGroundingUrls(null); // Clear previous URLs
    setLabsProofreadingResult(null); // Clear proofread result on new submission


    try {
      const prompt = `You are a patient advocate helping someone understand their lab results. Provide up-to-date and accurate information.
Analyze the provided information (text and/or image) to identify specific lab tests and their values. If no clear lab results are found, please state "No clear lab results found in the provided input."

Lab results: "${labsInput}"
(If an image is also provided, extract lab details from it.)

Please provide:
Overall Summary: [A concise 2-3 sentence summary of the main findings and implications of all the lab results together.]

For each test result identified, provide:
1. What it measures in simple terms
2. Whether it appears high, low, or normal (if obvious from typical ranges)
3. What it might mean (general information, not diagnosis)
4. What questions to ask the doctor

Format clearly with "Overall Summary:" followed by headers for each test. Use markdown for lists and emphasis.`; // Added markdown instruction

      const result = await executeGeminiCall(
        () => callGemini({ prompt, useSearchGrounding: true, imageFile: labsImageFile || undefined }), // Enable search grounding and image
        setLabsError
      );
      if (result) {
        setLabsOutput(result.text);
        setLabsGroundingUrls(result.groundingUrls || null); // Store URLs
      }
    } catch (error) {
      // Error already handled by executeGeminiCall
    } finally {
      setLabsLoading(false);
    }
  }, [labsInput, labsImageFile, executeGeminiCall, setLabsLoading, setLabsOutput, setLabsError, setLabsGroundingUrls, setLabsProofreadingResult]); // Added all state variables and setters to dependencies

  // Generic Proofread Handler
  const handleProofread = useCallback(async (
    currentInput: string,
    setProofreadingLoading: React.Dispatch<React.SetStateAction<boolean>>,
    setProofreadingError: React.Dispatch<React.SetStateAction<string | null>>,
    setProofreadingResult: React.Dispatch<React.SetStateAction<string | null>>
  ) => {
    if (!currentInput.trim()) {
      alert('Please enter text to proofread.');
      return;
    }
    setProofreadingLoading(true);
    setProofreadingResult(null);
    setProofreadingError(null);

    try {
      const result = await executeGeminiCall(
        () => callGeminiProofread(currentInput),
        setProofreadingError
      );
      if (result) { // Only set result if not undefined
        setProofreadingResult(result);
      }
    } catch (error) {
      // Error already handled by executeGeminiCall
    } finally {
      setProofreadingLoading(false);
    }
  }, [executeGeminiCall]); // Only executeGeminiCall is a dependency as others are function parameters.

  const handleProofreadTranslate = useCallback(() => {
    handleProofread(
      translateInput,
      setTranslateProofreadingLoading,
      setTranslateProofreadingError,
      setTranslateProofreadingResult
    );
  }, [translateInput, handleProofread, setTranslateProofreadingLoading, setTranslateProofreadingError, setTranslateProofreadingResult]); // Added all state variables and setters to dependencies.

  const handleProofreadPrepare = useCallback(() => {
    handleProofread(
      prepareInput,
      setPrepareProofreadingLoading,
      setPrepareProofreadingError,
      setPrepareProofreadingResult
    );
  }, [prepareInput, handleProofread, setPrepareProofreadingLoading, setPrepareProofreadingError, setPrepareProofreadingResult]); // Added all state variables and setters to dependencies.

  const handleProofreadSummarize = useCallback(() => {
    handleProofread(
      summarizeInput,
      setSummarizeProofreadingLoading,
      setSummarizeProofreadingError,
      setSummarizeProofreadingResult
    );
  }, [summarizeInput, handleProofread, setSummarizeProofreadingLoading, setSummarizeProofreadingError, setSummarizeProofreadingResult]); // Added all state variables and setters to dependencies.

  const handleProofreadLabs = useCallback(() => {
    handleProofread(
      labsInput,
      setLabsProofreadingLoading,
      setLabsProofreadingError,
      setLabsProofreadingResult
    );
  }, [labsInput, handleProofread, setLabsProofreadingLoading, setLabsProofreadingError, setLabsProofreadingResult]); // Added all state variables and setters to dependencies.


  // Live API Recording Functions
  const handleStartRecording = useCallback(async () => {
    setLiveLoading(true);
    setLiveError(null);
    setTranscriptionHistory([]);
    setCurrentLiveInputTranscription('');
    setCurrentLiveOutputTranscription('');
    // Fix: Clear refs when starting a new session
    latestInputTranscriptionRef.current = '';
    latestOutputTranscriptionRef.current = '';

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      // Fix: Ensure AudioContext is initialized with appropriate fallback for webkitAudioContext as per guidelines
      const inputAudioContext = new (window.AudioContext || window.webkitAudioContext!)({ sampleRate: 16000 });
      inputAudioContextRef.current = inputAudioContext;
      // Fix: Ensure AudioContext is initialized with appropriate fallback for webkitAudioContext as per guidelines
      const outputAudioContext = new (window.AudioContext || window.webkitAudioContext!)({ sampleRate: 24000 });
      outputAudioContextRef.current = outputAudioContext;

      // Initialize output node and connect it to the destination
      const outputNode = outputAudioContext.createGain();
      outputNode.connect(outputAudioContext.destination);
      outputNodeRef.current = outputNode;

      const source = inputAudioContext.createMediaStreamSource(stream);
      const scriptProcessor = inputAudioContext.createScriptProcessor(4096, 1, 1);
      scriptProcessorNodeRef.current = scriptProcessor;

      // Create new instance to ensure up-to-date API key
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY }); 

      liveSessionPromiseRef.current = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        callbacks: {
          onopen: () => {
            console.debug('Live session opened');
            setIsRecording(true);
            setLiveLoading(false);
            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              // CRITICAL: Solely rely on sessionPromise resolves and then call `session.sendRealtimeInput`, **do not** add other condition checks.
              liveSessionPromiseRef.current?.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioContext.destination);
          },
          onmessage: async (message: LiveServerMessage) => {
            // Audio output from model
            const base64EncodedAudioString = message.serverContent?.modelTurn?.parts[0]?.inlineData.data;
            if (base64EncodedAudioString) {
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputAudioContextRef.current!.currentTime);
              const audioBuffer = await decodeAudioData(
                decode(base64EncodedAudioString),
                outputAudioContextRef.current!,
                24000,
                1,
              );
              const audioSource = outputAudioContextRef.current!.createBufferSource();
              audioSource.buffer = audioBuffer;
              // Connect to the outputNode, which is already connected to the destination
              audioSource.connect(outputNodeRef.current!); 
              audioSource.addEventListener('ended', () => {
                audioSourcesRef.current.delete(audioSource);
              });

              audioSource.start(nextStartTimeRef.current);
              nextStartTimeRef.current = nextStartTimeRef.current + audioBuffer.duration;
              audioSourcesRef.current.add(audioSource);
            }

            // Transcription
            if (message.serverContent?.outputTranscription) {
              // Fix: Use the ref to update instead of state directly to avoid stale closures.
              latestOutputTranscriptionRef.current += message.serverContent!.outputTranscription!.text;
              setCurrentLiveOutputTranscription(latestOutputTranscriptionRef.current);
            }
            if (message.serverContent?.inputTranscription) {
              // Fix: Use the ref to update instead of state directly to avoid stale closures.
              latestInputTranscriptionRef.current += message.serverContent!.inputTranscription!.text;
              setCurrentLiveInputTranscription(latestInputTranscriptionRef.current);
            }

            // Turn complete
            if (message.serverContent?.turnComplete) {
              // Fix: Use the latest ref values for history to avoid stale closures
              const finalInputTranscription = latestInputTranscriptionRef.current;
              const finalOutputTranscription = latestOutputTranscriptionRef.current;

              if (finalInputTranscription) {
                setTranscriptionHistory(prev => [...prev, { speaker: 'User', text: finalInputTranscription.trim() }]);
                setCurrentLiveInputTranscription(''); // Clear for next turn's "current speaking" UI
                latestInputTranscriptionRef.current = ''; // Clear ref too
              }
              if (finalOutputTranscription) {
                setTranscriptionHistory(prev => [...prev, { speaker: 'AI', text: finalOutputTranscription.trim() }]);
                setCurrentLiveOutputTranscription(''); // Clear for next turn's "current speaking" UI
                latestOutputTranscriptionRef.current = ''; // Clear ref too
              }
            }
            
            // Interrupted
            const interrupted = message.serverContent?.interrupted;
            if (interrupted) {
              for (const sourceNode of audioSourcesRef.current.values()) {
                sourceNode.stop();
                audioSourcesRef.current.delete(sourceNode);
              }
              nextStartTimeRef.current = 0;
            }
          },
          onerror: (e: ErrorEvent) => {
            console.error('Live session error:', e);
            // Handle specific API key error for Live API
            if (e.message.includes("Requested entity was not found.") && window.aistudio) {
              setLiveError("Your API key might be invalid or expired for Live API. Please re-select it.");
              checkApiKeyStatus(); // Trigger re-check to show select key button
            } else {
              setLiveError(`Live session error: ${e.message}`);
            }
            // Fix: Added handleStopRecording to clean up after error
            handleStopRecording(); // Stop recording cleanly
          },
          onclose: (e: CloseEvent) => {
            console.debug('Live session closed:', e);
            // Don't call handleStopRecording here to avoid loop, just reset states
            setIsRecording(false);
            setLiveLoading(false);
            if (e.code !== 1000) { // 1000 is normal closure
              setLiveError(`Live session closed unexpectedly: Code ${e.code}`);
            }
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } }, // Use Zephyr voice
          },
          inputAudioTranscription: {}, // Enable transcription for user input
          outputAudioTranscription: {}, // Enable transcription for model output
          systemInstruction: 'You are a helpful patient advocate assistant during a medical consultation.',
        },
      });
      liveSessionRef.current = await liveSessionPromiseRef.current; // Store the resolved session

    } catch (error: any) {
      console.error('Error starting recording:', error);
      let errorMessage = error instanceof Error ? error.message : String(error);
      if (errorMessage.includes("Permission denied")) {
        errorMessage = "Microphone access denied. Please allow microphone usage to record.";
      } else if (errorMessage.includes("Gemini API key is not configured.") && window.aistudio) {
        errorMessage = "Gemini API key is not configured. Please select your API key again from the platform dialog.";
        await checkApiKeyStatus(); // Trigger re-check with await
      }
      setLiveError(`Failed to start recording: ${errorMessage}`);
      setLiveLoading(false);
      setIsRecording(false);
    }
  }, [
    checkApiKeyStatus,
    handleStopRecording, // This needs to be a dependency
    mediaStreamRef,
    inputAudioContextRef,
    outputAudioContextRef,
    outputNodeRef,
    scriptProcessorNodeRef,
    liveSessionPromiseRef,
    liveSessionRef,
    nextStartTimeRef,
    audioSourcesRef,
    latestInputTranscriptionRef,
    latestOutputTranscriptionRef,
    setLiveLoading,
    setLiveError,
    setTranscriptionHistory,
    setCurrentLiveInputTranscription,
    setCurrentLiveOutputTranscription,
    setIsRecording,
  ]);

  // Cleanup effect
  useEffect(() => {
    return () => {
      // Ensure everything is stopped if component unmounts
      handleStopRecording();
    };
  }, [handleStopRecording]); // handleStopRecording is a useCallback, so must be dependency.


  return (
    <div className="max-w-4xl mx-auto bg-white rounded-2xl shadow-2xl overflow-hidden">
      <header className="bg-gradient-to-br from-indigo-600 to-purple-800 text-white p-8 text-center">
        <h1 className="text-4xl font-bold mb-2 flex items-center justify-center gap-3">
          🩺 Patient Advocate AI
        </h1>
        <p className="opacity-90 text-lg">
          Your personal medical advocate - powered by Gemini AI
        </p>
      </header>

      {/* API Key Status Banner */}
      <div className={`p-5 m-5 rounded-xl text-center ${apiConfigured ? 'bg-green-100 border-2 border-success text-green-800' : 'bg-orange-100 border-2 border-warning text-orange-800'}`}>
        <h3 className="text-xl font-semibold text-amber-700 mb-3">
          {apiConfigured ? '✅ Gemini AI Ready' : '🔑 API Key Status'}
        </h3>
        <p className="mb-4">{apiKeyStatusMessage}</p>
        {showSelectKeyButton && (
          <>
            <button
              className="bg-primary text-white py-2 px-4 rounded-lg text-base font-semibold transition-all hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed mt-3 mr-2"
              onClick={handleSelectApiKey}
              disabled={apiIsSelecting}
            >
              {apiIsSelecting ? (
                  <Spinner size="w-5 h-5" color="border-t-white" className="inline-block mr-2" />
                ) : (
                  'Select Gemini API Key'
                )}
            </button>
            <small className="block mt-4 text-amber-900">
              Billing details:{' '}
              <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">ai.google.dev/gemini-api/docs/billing</a>
            </small>
          </>
        )}
        {!apiConfigured && !showSelectKeyButton && ( // Show only if aistudio not available or hasSelectedKey failed and no button is shown
           <small className="block mt-4 text-amber-900">
            Don't have an API key? Get one free at{' '}
            <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">Google AI Studio</a>.
            For local development, please set the <code>API_KEY</code> environment variable.
          </small>
        )}
      </div>

      <nav className="flex border-b-2 border-borderColor overflow-x-auto">
        <button
          className={`flex-1 px-4 py-4 border-b-4 border-transparent text-textSecondary font-semibold text-sm transition-all hover:bg-bgSecondary hover:text-primary ${
            activeTab === 'translate' ? 'border-primary text-primary bg-bgSecondary' : ''
          }`}
          onClick={() => handleTabChange('translate')}
        >
          📝 Translate Jargon
        </button>
        <button
          className={`flex-1 px-4 py-4 border-b-4 border-transparent text-textSecondary font-semibold text-sm transition-all hover:bg-bgSecondary hover:text-primary ${
            activeTab === 'prepare' ? 'border-primary text-primary bg-bgSecondary' : ''
          }`}
          onClick={() => handleTabChange('prepare')}
        >
          📋 Prep Appointment
        </button>
        <button
          className={`flex-1 px-4 py-4 border-b-4 border-transparent text-textSecondary font-semibold text-sm transition-all hover:bg-bgSecondary hover:text-primary ${
            activeTab === 'summarize' ? 'border-primary text-primary bg-bgSecondary' : ''
          }`}
          onClick={() => handleTabChange('summarize')}
        >
          💊 Summarize Notes
        </button>
        <button
          className={`flex-1 px-4 py-4 border-b-4 border-transparent text-textSecondary font-semibold text-sm transition-all hover:bg-bgSecondary hover:text-primary ${
            activeTab === 'labs' ? 'border-primary text-primary bg-bgSecondary' : ''
          }`}
          onClick={() => handleTabChange('labs')}
        >
          🔬 Explain Labs
        </button>
        <button
          className={`flex-1 px-4 py-4 border-b-4 border-transparent text-textSecondary font-semibold text-sm transition-all hover:bg-bgSecondary hover:text-primary ${
            activeTab === 'record' ? 'border-primary text-primary bg-bgSecondary' : ''
          }`}
          onClick={() => handleTabChange('record')}
          disabled={!apiConfigured}
        >
          🎙️ Record Session
        </button>
      </nav>

      <main className="p-6">
        {/* Translate Jargon Tab */}
        {activeTab === 'translate' && (
          <section>
            <h2 className="text-2xl font-bold text-primary mb-5">Translate Medical Jargon</h2>
            <Disclaimer>
              This tool provides general information and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.
            </Disclaimer>

            <div className="mt-6 mb-4">
              <label htmlFor="translate-input" className="block text-lg font-medium text-textPrimary mb-2">
                Paste medical text here:
              </label>
              <textarea
                id="translate-input"
                className="w-full p-3 border border-borderColor rounded-lg focus:ring focus:ring-primary focus:border-primary transition-all text-textPrimary"
                rows={6}
                value={translateInput}
                onChange={(e) => setTranslateInput(e.target.value)}
                placeholder="e.g., 'The patient presents with dysphagia and odynophagia, requiring an EGD with biopsies to rule out eosinophilic esophagitis.'"
              ></textarea>
              <button
                className="bg-primary text-white py-2 px-5 rounded-lg text-base font-semibold transition-all hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed mt-3 mr-2"
                onClick={handleTranslateJargon}
                disabled={translateLoading || !apiConfigured}
              >
                {translateLoading ? (
                  <Spinner size="w-5 h-5" color="border-t-white" className="inline-block mr-2" />
                ) : (
                  'Translate'
                )}
              </button>
              <button
                className="bg-gray-200 text-textSecondary py-2 px-5 rounded-lg text-base font-semibold transition-all hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed mt-3"
                onClick={handleProofreadTranslate}
                disabled={translateProofreadingLoading || !translateInput.trim()}
              >
                {translateProofreadingLoading ? (
                  <Spinner size="w-5 h-5" color="border-t-gray-700" className="inline-block mr-2" />
                ) : (
                  'Proofread Input'
                )}
              </button>
            </div>

            {translateError && <ErrorMessage message={translateError} />}
            {translateOutput && (
              <ResultSection
                title="Simplified Explanation & Questions"
                content={renderMarkdown(translateOutput)}
              />
            )}
            {translateProofreadingError && <ErrorMessage message={translateProofreadingError} />}
            {translateProofreadingResult && translateProofreadingResult !== translateInput && (
              <ResultSection
                title="Proofread Input"
                content={renderMarkdown(translateProofreadingResult)}
                variant="info"
              />
            )}
            {translateProofreadingResult && translateProofreadingResult === translateInput && !translateProofreadingLoading && (
              <ResultSection
                title="Proofread Input"
                content={<p>No corrections needed.</p>}
                variant="info"
              />
            )}
          </section>
        )}

        {/* Prepare for Appointment Tab */}
        {activeTab === 'prepare' && (
          <section>
            <h2 className="text-2xl font-bold text-primary mb-5">Prepare for Appointment</h2>
            <Disclaimer>
              This tool provides general information and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.
            </Disclaimer>

            <div className="mt-6 mb-4">
              <label htmlFor="prepare-input" className="block text-lg font-medium text-textPrimary mb-2">
                Describe your symptoms and concerns:
              </label>
              <textarea
                id="prepare-input"
                className="w-full p-3 border border-borderColor rounded-lg focus:ring focus:ring-primary focus:border-primary transition-all text-textPrimary"
                rows={6}
                value={prepareInput}
                onChange={(e) => setPrepareInput(e.target.value)}
                placeholder="e.g., 'I've had a persistent cough for two weeks, accompanied by a low-grade fever and fatigue. I'm worried about bronchitis or pneumonia.'"
              ></textarea>
              <button
                className="bg-primary text-white py-2 px-5 rounded-lg text-base font-semibold transition-all hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed mt-3 mr-2"
                onClick={handlePrepareAppointment}
                disabled={prepareLoading || !apiConfigured}
              >
                {prepareLoading ? (
                  <Spinner size="w-5 h-5" color="border-t-white" className="inline-block mr-2" />
                ) : (
                  'Generate Guide'
                )}
              </button>
              <button
                className="bg-gray-200 text-textSecondary py-2 px-5 rounded-lg text-base font-semibold transition-all hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed mt-3"
                onClick={handleProofreadPrepare}
                disabled={prepareProofreadingLoading || !prepareInput.trim()}
              >
                {prepareProofreadingLoading ? (
                  <Spinner size="w-5 h-5" color="border-t-gray-700" className="inline-block mr-2" />
                ) : (
                  'Proofread Input'
                )}
              </button>
            </div>

            {prepareError && <ErrorMessage message={prepareError} />}
            {prepareOutput && (
              <ResultSection
                title="Appointment Preparation Guide"
                content={renderMarkdown(prepareOutput)}
              />
            )}
            {prepareGroundingUrls && prepareGroundingUrls.length > 0 && (
              <ResultSection
                title="Relevant Information from Google Search"
                variant="info"
                content={
                  <ul className="list-disc pl-5">
                    {prepareGroundingUrls.map((link, index) => (
                      <li key={index} className="mb-1">
                        <a href={link.uri} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                          {link.title || link.uri}
                        </a>
                      </li>
                    ))}
                  </ul>
                }
              />
            )}
            {prepareProofreadingError && <ErrorMessage message={prepareProofreadingError} />}
            {prepareProofreadingResult && prepareProofreadingResult !== prepareInput && (
              <ResultSection
                title="Proofread Input"
                content={renderMarkdown(prepareProofreadingResult)}
                variant="info"
              />
            )}
            {prepareProofreadingResult && prepareProofreadingResult === prepareInput && !prepareProofreadingLoading && (
              <ResultSection
                title="Proofread Input"
                content={<p>No corrections needed.</p>}
                variant="info"
              />
            )}
          </section>
        )}

        {/* Summarize Notes Tab */}
        {activeTab === 'summarize' && (
          <section>
            <h2 className="text-2xl font-bold text-primary mb-5">Summarize Appointment Notes / Prescriptions</h2>
            <Disclaimer>
              This tool provides general information and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment.
            </Disclaimer>

            <div className="mt-6 mb-4">
              <label htmlFor="summarize-input" className="block text-lg font-medium text-textPrimary mb-2">
                Paste your notes or prescription text:
              </label>
              <textarea
                id="summarize-input"
                className="w-full p-3 border border-borderColor rounded-lg focus:ring focus:ring-primary focus:border-primary transition-all text-textPrimary"
                rows={6}
                value={summarizeInput}
                onChange={(e) => setSummarizeInput(e.target.value)}
                placeholder="e.g., 'Patient seen for follow-up on hypertension. BP 140/90. Started on Lisinopril 10mg QD. Follow up in 4 weeks. Avoid high sodium diet.'"
              ></textarea>

              <label htmlFor="summarize-image-input" className="block text-lg font-medium text-textPrimary mt-4 mb-2">
                Or upload an image of your notes/prescription (optional):
              </label>
              <input
                id="summarize-image-input"
                type="file"
                accept="image/*"
                onChange={handleSummarizeImageChange}
                className="block w-full text-sm text-textPrimary
                           file:mr-4 file:py-2 file:px-4
                           file:rounded-full file:border-0
                           file:text-sm file:font-semibold
                           file:bg-violet-50 file:text-primary
                           hover:file:bg-violet-100 mb-3"
              />
              {summarizeImagePreviewUrl && (
                <div className="mt-2 relative w-48 h-48 border border-borderColor rounded-lg overflow-hidden">
                  <img src={summarizeImagePreviewUrl} alt="Image preview" className="object-cover w-full h-full" />
                  <button
                    onClick={handleClearSummarizeImage}
                    className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 text-xs"
                    aria-label="Remove image"
                  >
                    ✕
                  </button>
                </div>
              )}

              <button
                className="bg-primary text-white py-2 px-5 rounded-lg text-base font-semibold transition-all hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed mt-3 mr-2"
                onClick={handleSummarizeNotes}
                disabled={summarizeLoading || (!summarizeInput.trim() && !summarizeImageFile) || !apiConfigured}
              >
                {summarizeLoading ? (
                  <Spinner size="w-5 h-5" color="border-t-white" className="inline-block mr-2" />
                ) : (
                  'Summarize Notes'
                )}
              </button>
              <button
                className="bg-gray-200 text-textSecondary py-2 px-5 rounded-lg text-base font-semibold transition-all hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed mt-3"
                onClick={handleProofreadSummarize}
                disabled={summarizeProofreadingLoading || !summarizeInput.trim()}
              >
                {summarizeProofreadingLoading ? (
                  <Spinner size="w-5 h-5" color="border-t-gray-700" className="inline-block mr-2" />
                ) : (
                  'Proofread Input'
                )}
              </button>
            </div>

            {summarizeError && <ErrorMessage message={summarizeError} />}
            {summarizeOutput && (
              <ResultSection
                title="Notes Summary & Action Items"
                content={renderMarkdown(summarizeOutput)}
              />
            )}
            {extractedMedications && (
              <ResultSection
                title="Medication Details"
                content={renderMarkdown(extractedMedications)}
                variant="info"
              />
            )}
            {summarizeProofreadingError && <ErrorMessage message={summarizeProofreadingError} />}
            {summarizeProofreadingResult && summarizeProofreadingResult !== summarizeInput && (
              <ResultSection
                title="Proofread Input"
                content={renderMarkdown(summarizeProofreadingResult)}
                variant="info"
              />
            )}
            {summarizeProofreadingResult && summarizeProofreadingResult === summarizeInput && !summarizeProofreadingLoading && (
              <ResultSection
                title="Proofread Input"
                content={<p>No corrections needed.</p>}
                variant="info"
              />
            )}
          </section>
        )}

        {/* Explain Lab Results Tab */}
        {activeTab === 'labs' && (
          <section>
            <h2 className="text-2xl font-bold text-primary mb-5">Explain Lab Results</h2>
            <Disclaimer>
              This tool provides general information and should not replace professional medical advice. Always consult with a qualified healthcare provider for diagnosis and treatment. Lab results interpretation is complex and dependent on individual factors.
            </Disclaimer>

            <div className="mt-6 mb-4">
              <label htmlFor="labs-input" className="block text-lg font-medium text-textPrimary mb-2">
                Paste your lab results text:
              </label>
              <textarea
                id="labs-input"
                className="w-full p-3 border border-borderColor rounded-lg focus:ring focus:ring-primary focus:border-primary transition-all text-textPrimary"
                rows={6}
                value={labsInput}
                onChange={(e) => setLabsInput(e.target.value)}
                placeholder="e.g., 'Glucose: 120 mg/dL (Normal: 70-99 mg/dL), Cholesterol Total: 220 mg/dL (Normal: <200 mg/dL), HDL: 40 mg/dL (Normal: >60 mg/dL)'"
              ></textarea>

              <label htmlFor="labs-image-input" className="block text-lg font-medium text-textPrimary mt-4 mb-2">
                Or upload an image of your lab results (optional):
              </label>
              <input
                id="labs-image-input"
                type="file"
                accept="image/*"
                onChange={handleLabsImageChange}
                className="block w-full text-sm text-textPrimary
                           file:mr-4 file:py-2 file:px-4
                           file:rounded-full file:border-0
                           file:text-sm file:font-semibold
                           file:bg-violet-50 file:text-primary
                           hover:file:bg-violet-100 mb-3"
              />
              {labsImagePreviewUrl && (
                <div className="mt-2 relative w-48 h-48 border border-borderColor rounded-lg overflow-hidden">
                  <img src={labsImagePreviewUrl} alt="Image preview" className="object-cover w-full h-full" />
                  <button
                    onClick={handleClearLabsImage}
                    className="absolute top-1 right-1 bg-red-500 text-white rounded-full p-1 text-xs"
                    aria-label="Remove image"
                  >
                    ✕
                  </button>
                </div>
              )}

              <button
                className="bg-primary text-white py-2 px-5 rounded-lg text-base font-semibold transition-all hover:bg-primary-hover disabled:opacity-50 disabled:cursor-not-allowed mt-3 mr-2"
                onClick={handleExplainLabs}
                disabled={labsLoading || (!labsInput.trim() && !labsImageFile) || !apiConfigured}
              >
                {labsLoading ? (
                  <Spinner size="w-5 h-5" color="border-t-white" className="inline-block mr-2" />
                ) : (
                  'Explain Results'
                )}
              </button>
              <button
                className="bg-gray-200 text-textSecondary py-2 px-5 rounded-lg text-base font-semibold transition-all hover:bg-gray-300 disabled:opacity-50 disabled:cursor-not-allowed mt-3"
                onClick={handleProofreadLabs}
                disabled={labsProofreadingLoading || !labsInput.trim()}
              >
                {labsProofreadingLoading ? (
                  <Spinner size="w-5 h-5" color="border-t-gray-700" className="inline-block mr-2" />
                ) : (
                  'Proofread Input'
                )}
              </button>
            </div>

            {labsError && <ErrorMessage message={labsError} />}
            {labsOutput && (
              <ResultSection
                title="Lab Results Explanation"
                content={renderMarkdown(labsOutput)}
              />
            )}
            {labsGroundingUrls && labsGroundingUrls.length > 0 && (
              <ResultSection
                title="Relevant Information from Google Search"
                variant="info"
                content={
                  <ul className="list-disc pl-5">
                    {labsGroundingUrls.map((link, index) => (
                      <li key={index} className="mb-1">
                        <a href={link.uri} target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                          {link.title || link.uri}
                        </a>
                      </li>
                    ))}
                  </ul>
                }
              />
            )}
            {labsProofreadingError && <ErrorMessage message={labsProofreadingError} />}
            {labsProofreadingResult && labsProofreadingResult !== labsInput && (
              <ResultSection
                title="Proofread Input"
                content={renderMarkdown(labsProofreadingResult)}
                variant="info"
              />
            )}
            {labsProofreadingResult && labsProofreadingResult === labsInput && !labsProofreadingLoading && (
              <ResultSection
                title="Proofread Input"
                content={<p>No corrections needed.</p>}
                variant="info"
              />
            )}
          </section>
        )}

        {/* Record Session Tab */}
        {activeTab === 'record' && (
          <section>
            <h2 className="text-2xl font-bold text-primary mb-5">Record Session</h2>
            <Disclaimer>
              This tool provides real-time AI assistance during conversations. It captures audio and processes it with AI. Ensure you have explicit consent from all parties before recording any conversation. This tool should not replace professional medical advice.
            </Disclaimer>

            <div className="mt-6 mb-4 flex justify-center">
              {!isRecording ? (
                <button
                  className="bg-green-600 text-white py-3 px-6 rounded-lg text-lg font-semibold transition-all hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                  onClick={handleStartRecording}
                  disabled={liveLoading || !apiConfigured}
                >
                  {liveLoading ? (
                    <Spinner size="w-6 h-6" color="border-t-white" className="inline-block mr-2" />
                  ) : (
                    <>
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
                      </svg>
                      Start Recording
                    </>
                  )}
                </button>
              ) : (
                <button
                  className="bg-red-600 text-white py-3 px-6 rounded-lg text-lg font-semibold transition-all hover:bg-red-700 flex items-center gap-2"
                  onClick={handleStopRecording}
                  disabled={liveLoading} // Disable if still in a loading/closing state
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z" />
                  </svg>
                  Stop Recording
                </button>
              )}
            </div>

            {liveError && <ErrorMessage message={liveError} />}
            {liveLoading && !isRecording && <ResultSection title="Starting Session..." content="Please wait while the AI prepares. You may be prompted for microphone access." />}

            <div className="mt-8">
              <h3 className="text-xl font-semibold text-primary mb-4">Conversation Transcript:</h3>
              <div className="bg-bgSecondary p-4 rounded-lg h-96 overflow-y-auto border border-borderColor">
                {transcriptionHistory.length === 0 && currentLiveInputTranscription === '' && currentLiveOutputTranscription === '' && (
                  <p className="text-textSecondary italic">Start recording to see the transcript.</p>
                )}
                {transcriptionHistory.map((entry, index) => (
                  <p key={index} className={`mb-2 ${entry.speaker === 'User' ? 'text-gray-800' : 'text-primary'}`}>
                    <strong>{entry.speaker}:</strong> {entry.text}
                  </p>
                ))}
                {currentLiveInputTranscription && (
                  <p className="mb-2 text-gray-700"><strong>User (speaking):</strong> {currentLiveInputTranscription}</p>
                )}
                {currentLiveOutputTranscription && (
                  <p className="mb-2 text-primary"><strong>AI (speaking):</strong> {currentLiveOutputTranscription}</p>
                )}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
};

export default App;
