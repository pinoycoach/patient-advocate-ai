
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
    ai?: { // Minimal declaration for proofread to work with Chrome AI
      proofread: (options: { text: string }) => Promise<{ text: string }>;
    };
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
      setTranscriptionHistory(prev => [...prev, { speaker: 'Model', text: latestOutputTranscriptionRef.current.trim() }]);
      setCurrentLiveOutputTranscription('');
      latestOutputTranscriptionRef.current = ''; // Clear ref too
    }
  }, []); // No dependencies needed for a full stop/reset.

  // Reusable proofreading logic
  const proofreadText = useCallback(async ({
    textToProofread,
    setLoading,
    setError,
    setResult,
  }: {
    textToProofread: string;
    setLoading: React.Dispatch<React.SetStateAction<boolean>>;
    setError: React.Dispatch<React.SetStateAction<string | null>>;
    setResult: React.Dispatch<React.SetStateAction<string | null>>;
  }) => {
    if (!textToProofread.trim()) {
      alert('Please enter text to proofread.');
      return;
    }

    setLoading(true);
    setResult(null);
    setError(null);

    // --- START of Chrome AI Proofreader API Integration ---
    if (window.ai && 'proofread' in window.ai) {
      try {
        const result = await window.ai.proofread({ text: textToProofread });
        setResult(result.text);
      } catch (error) {
        console.error('Chrome AI Proofread Error:', error);
        setError('Proofreading failed using Chrome AI. Ensure Chrome AI features are enabled and permissions are granted.');
      } finally {
        setLoading(false);
      }
    } else {
      // --- Fallback to existing Gemini Proofread logic ---
      try {
        const result = await callGeminiProofread(textToProofread);
        if (result) {
          setResult(result);
        }
      } catch (error) {
        console.error('Gemini Proofread Error:', error);
        setError(error instanceof Error ? error.message : 'An unknown error occurred during Gemini proofreading.');
      } finally {
        setLoading(false);
      }
      // --- END of Fallback ---
    }
  }, []); // Dependencies are stable, so empty array is fine.


  const handleStartRecording = useCallback(async () => {
    // Check if API key is configured before starting recording
    if (!apiConfigured) {
      // Offer to select key if not configured
      setShowSelectKeyButton(true);
      setApiKeyStatusMessage('Please select an API key to enable the recording feature.');
      return;
    }

    setLiveLoading(true);
    setLiveError(null);
    setIsRecording(true);
    setTranscriptionHistory([]);
    setCurrentLiveInputTranscription('');
    setCurrentLiveOutputTranscription('');
    latestInputTranscriptionRef.current = '';
    latestOutputTranscriptionRef.current = '';


    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      inputAudioContextRef.current = new AudioContextClass({ sampleRate: 16000 });
      outputAudioContextRef.current = new AudioContextClass({ sampleRate: 24000 });
      outputNodeRef.current = outputAudioContextRef.current.createGain(); // Initialize outputNodeRef
      outputNodeRef.current.connect(outputAudioContextRef.current.destination);

      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY }); // Create new instance for up-to-date key

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-09-2025',
        callbacks: {
          onopen: () => {
            console.debug('Live session opened');
            const source = inputAudioContextRef.current!.createMediaStreamSource(stream);
            const scriptProcessor = inputAudioContextRef.current!.createScriptProcessor(4096, 1, 1);
            scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
              const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
              const pcmBlob = createBlob(inputData);
              sessionPromise.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };
            source.connect(scriptProcessor);
            scriptProcessor.connect(inputAudioContextRef.current!.destination);
            scriptProcessorNodeRef.current = scriptProcessor;
          },
          onmessage: async (message: LiveServerMessage) => {
            // Audio output processing
            const base64EncodedAudioString = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64EncodedAudioString) {
              nextStartTimeRef.current = Math.max(
                nextStartTimeRef.current,
                outputAudioContextRef.current!.currentTime,
              );
              const audioBuffer = await decodeAudioData(
                decode(base64EncodedAudioString),
                outputAudioContextRef.current!,
                24000,
                1,
              );
              const source = outputAudioContextRef.current!.createBufferSource();
              source.buffer = audioBuffer;
              source.connect(outputNodeRef.current!);
              source.addEventListener('ended', () => {
                audioSourcesRef.current.delete(source);
              });

              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current = nextStartTimeRef.current + audioBuffer.duration;
              audioSourcesRef.current.add(source);
            }

            const interrupted = message.serverContent?.interrupted;
            if (interrupted) {
              for (const source of audioSourcesRef.current.values()) {
                source.stop();
                audioSourcesRef.current.delete(source);
              }
              nextStartTimeRef.current = 0;
            }

            // Transcription processing
            if (message.serverContent?.outputTranscription) {
              latestOutputTranscriptionRef.current += message.serverContent.outputTranscription.text;
              setCurrentLiveOutputTranscription(latestOutputTranscriptionRef.current);
            } else if (message.serverContent?.inputTranscription) {
              latestInputTranscriptionRef.current += message.serverContent.inputTranscription.text;
              setCurrentLiveInputTranscription(latestInputTranscriptionRef.current);
            }

            if (message.serverContent?.turnComplete) {
              const fullInputTranscription = latestInputTranscriptionRef.current.trim();
              const fullOutputTranscription = latestOutputTranscriptionRef.current.trim();

              if (fullInputTranscription) {
                setTranscriptionHistory(prev => [...prev, { speaker: 'User', text: fullInputTranscription }]);
              }
              if (fullOutputTranscription) {
                setTranscriptionHistory(prev => [...prev, { speaker: 'Model', text: fullOutputTranscription }]);
              }

              latestInputTranscriptionRef.current = '';
              latestOutputTranscriptionRef.current = '';
              setCurrentLiveInputTranscription('');
              setCurrentLiveOutputTranscription('');
            }

            // Handle function calls (if any, though not implemented in this app yet)
            if (message.toolCall) {
              console.log('Function call received:', message.toolCall);
              // Implement logic to execute function and send tool response if needed.
            }

          },
          onerror: (e: ErrorEvent) => {
            console.error('Live session error:', e);
            setLiveError('Live conversation encountered an error: ' + e.message);
            handleStopRecording();
          },
          onclose: (e: CloseEvent) => {
            console.debug('Live session closed:', e);
            if (!e.wasClean) {
              setLiveError('Live conversation closed unexpectedly.');
            }
            handleStopRecording();
          },
        },
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
          },
          systemInstruction: 'You are a friendly and helpful medical advocate. You explain medical terms, help prepare for appointments, summarize notes, and explain lab results.',
          inputAudioTranscription: {}, // Enable transcription for user input
          outputAudioTranscription: {}, // Enable transcription for model output
        },
      });
      liveSessionPromiseRef.current = sessionPromise;
      liveSessionRef.current = await sessionPromise;

    } catch (err: any) {
      console.error('Error starting live recording:', err);
      setLiveError('Failed to start recording: ' + err.message);
      setIsRecording(false);
      setLiveLoading(false);
      handleStopRecording(); // Ensure all resources are cleaned up
    }
  }, [apiConfigured, handleStopRecording]); // handleStopRecording is a dependency

  // Generic function to execute Gemini API calls and handle common loading/error states
  const executeGeminiCall = useCallback(async (
    apiCall: () => Promise<GeminiResponseData | string | undefined>,
    setErrorState: React.Dispatch<React.SetStateAction<string | null>>
  ): Promise<GeminiResponseData | string | undefined> => {
    setErrorState(null); // Clear previous errors
    if (!apiConfigured) {
      setShowSelectKeyButton(true);
      setApiKeyStatusMessage('Please select an API key to enable this feature.');
      return undefined;
    }
    try {
      return await apiCall();
    } catch (err: any) {
      console.error("API Call Error:", err);
      setErrorState(err.message || "An unexpected error occurred.");
      // Check if the error indicates a missing API key for Veo models (billing link applies)
      if (typeof err.message === 'string' && err.message.includes('Requested entity was not found.')) {
        setShowSelectKeyButton(true);
        setApiKeyStatusMessage('There was an issue with the API key. Please re-select your API key. (A link to billing documentation is provided in the key selection dialog.)');
      }
      return undefined; // Indicate failure
    }
  }, [apiConfigured]);


  // Effect to check API key status on component mount
  useEffect(() => {
    async function checkApiKey() {
      if (window.aistudio) {
        const hasKey = await window.aistudio.hasSelectedApiKey();
        setApiConfigured(hasKey);
        if (!hasKey) {
          setShowSelectKeyButton(true);
          setApiKeyStatusMessage('Please select your Gemini API key to use the app.');
        } else {
          setApiKeyStatusMessage('API Key is configured.');
        }
      } else {
        // Fallback for environments without window.aistudio (e.g., pure web)
        if (process.env.API_KEY) {
          setApiConfigured(true);
          setApiKeyStatusMessage('API Key is configured from environment variable.');
        } else {
          setApiKeyStatusMessage('API Key is not configured. Please set API_KEY environment variable.');
          setShowSelectKeyButton(false); // No select key button if aistudio isn't present
        }
      }
    }
    checkApiKey();
  }, []);

  const handleSelectApiKey = useCallback(async () => {
    if (window.aistudio) {
      setApiIsSelecting(true);
      setApiKeyStatusMessage('Opening API key selection dialog...');
      try {
        await window.aistudio.openSelectKey();
        // Assuming key selection was successful; re-check on next API call
        setApiConfigured(true);
        setShowSelectKeyButton(false);
        setApiKeyStatusMessage('API key selected. You can now use the features.');
      } catch (error) {
        console.error('Error opening API key selection:', error);
        setApiKeyStatusMessage('Failed to open API key selection dialog.');
      } finally {
        setApiIsSelecting(false);
      }
    } else {
      alert("This environment does not support `window.aistudio.openSelectKey()` for API key selection.");
    }
  }, []);

  const handleTranslateJargon = useCallback(async () => {
    setTranslateLoading(true);
    setTranslateOutput(null);
    setTranslateProofreadingResult(null); // Clear proofreading results on new translation
    const result = await executeGeminiCall(
      () => callGemini({
        prompt: `Translate the following medical jargon into simple, easy-to-understand language for a patient. Maintain the core meaning but use analogies or common terms where appropriate. If the text is already simple, just rephrase it slightly to sound even more natural without over-simplifying if complexity is necessary.\n\nMedical Jargon: "${translateInput}"`,
        systemInstruction: "You are a friendly, empathetic medical advocate, skilled at simplifying complex medical information.",
      }),
      setTranslateError
    );

    if (result) {
      setTranslateOutput(typeof result === 'string' ? result : result.text);
    }
    setTranslateLoading(false);
  }, [translateInput, executeGeminiCall]);

  const handlePrepareAppointment = useCallback(async () => {
    setPrepareLoading(true);
    setPrepareOutput(null);
    setPrepareGroundingUrls(null);
    setPrepareProofreadingResult(null); // Clear proofreading results
    const result = await executeGeminiCall(
      () => callGemini({
        prompt: `I am preparing for an appointment and I want to make sure I cover all my concerns. Here are my notes:\n\n"${prepareInput}"\n\nPlease help me organize these notes, suggest questions I should ask my doctor based on them, and highlight any important points I should definitely mention. Provide information grounded by Google Search if applicable.`,
        systemInstruction: "You are a helpful medical advocate assisting patients in preparing for doctor's appointments.",
        useSearchGrounding: true,
      }),
      setPrepareError
    );

    if (result) {
      setPrepareOutput(typeof result === 'string' ? result : result.text);
      if (typeof result !== 'string' && result.groundingUrls) {
        setPrepareGroundingUrls(result.groundingUrls);
      }
    }
    setPrepareLoading(false);
  }, [prepareInput, executeGeminiCall]);

  const handleSummarizeNotes = useCallback(async () => {
    setSummarizeLoading(true);
    setSummarizeOutput(null);
    setExtractedMedications(null);
    setSummarizeProofreadingResult(null); // Clear proofreading results
    let prompt: string;
    let systemInstruction: string;

    if (summarizeInput.trim() && summarizeImageFile) {
      prompt = `Summarize the following medical notes and extract any mentioned medications or treatment plans. Then list all medications under a separate heading "Medications:". Combine information from the text and the image.

Notes: "${summarizeInput}"`;
      systemInstruction = "You are a meticulous medical assistant, capable of summarizing notes and identifying key medical details from both text and visual input.";
    } else if (summarizeInput.trim()) {
      prompt = `Summarize the following medical notes and extract any mentioned medications or treatment plans. Then list all medications under a separate heading "Medications:".

Notes: "${summarizeInput}"`;
      systemInstruction = "You are a meticulous medical assistant, capable of summarizing notes and identifying key medical details.";
    } else if (summarizeImageFile) {
      prompt = `Summarize the medical notes provided in the image and extract any mentioned medications or treatment plans. Then list all medications under a separate heading "Medications:".`;
      systemInstruction = "You are a meticulous medical assistant, capable of summarizing notes and identifying key medical details from visual input.";
    } else {
      alert('Please enter some text or upload an image to summarize.');
      setSummarizeLoading(false);
      return;
    }

    const result = await executeGeminiCall(
      () => callGemini({
        prompt,
        systemInstruction,
        imageFile: summarizeImageFile,
      }),
      setSummarizeError
    );

    if (result) {
      const fullText = typeof result === 'string' ? result : result.text;
      const medicationRegex = /Medications:\s*([\s\S]*)/i;
      const medicationMatch = fullText.match(medicationRegex);

      if (medicationMatch && medicationMatch[1]) {
        setExtractedMedications(medicationMatch[1].trim());
        setSummarizeOutput(fullText.replace(medicationRegex, '').trim());
      } else {
        setSummarizeOutput(fullText);
        setExtractedMedications(null);
      }
    }
    setSummarizeLoading(false);
  }, [summarizeInput, summarizeImageFile, executeGeminiCall]);

  const handleLabsExplanation = useCallback(async () => {
    setLabsLoading(true);
    setLabsOutput(null);
    setLabsGroundingUrls(null);
    setLabsProofreadingResult(null); // Clear proofreading results
    let prompt: string;
    let systemInstruction: string;

    if (labsInput.trim() && labsImageFile) {
      prompt = `Explain the following lab results in simple terms for a patient, highlighting what is normal, what is abnormal, and what it might mean. Combine information from the text and the image. Use Google Search for additional context if necessary.

Lab Results: "${labsInput}"`;
      systemInstruction = "You are a friendly medical advocate, skilled at explaining lab results clearly and empathetically.";
    } else if (labsInput.trim()) {
      prompt = `Explain the following lab results in simple terms for a patient, highlighting what is normal, what is abnormal, and what it might mean. Use Google Search for additional context if necessary.

Lab Results: "${labsInput}"`;
      systemInstruction = "You are a friendly medical advocate, skilled at explaining lab results clearly and empathetically.";
    } else if (labsImageFile) {
      prompt = `Explain the lab results provided in the image in simple terms for a patient, highlighting what is normal, what is abnormal, and what it might mean. Use Google Search for additional context if necessary.`;
      systemInstruction = "You are a friendly medical advocate, skilled at explaining lab results clearly and empathetically from visual input.";
    } else {
      alert('Please enter some text or upload an image of lab results.');
      setLabsLoading(false);
      return;
    }

    const result = await executeGeminiCall(
      () => callGemini({
        prompt,
        systemInstruction,
        useSearchGrounding: true,
        imageFile: labsImageFile,
      }),
      setLabsError
    );

    if (result) {
      setLabsOutput(typeof result === 'string' ? result : result.text);
      if (typeof result !== 'string' && result.groundingUrls) {
        setLabsGroundingUrls(result.groundingUrls);
      }
    }
    setLabsLoading(false);
  }, [labsInput, labsImageFile, executeGeminiCall]);

  // Handlers for image file changes
  const handleSummarizeImageChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setSummarizeImageFile(file);
      setSummarizeImagePreviewUrl(URL.createObjectURL(file));
    } else {
      setSummarizeImageFile(null);
      setSummarizeImagePreviewUrl(null);
    }
  }, []);

  const handleLabsImageChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setLabsImageFile(file);
      setLabsImagePreviewUrl(URL.createObjectURL(file));
    } else {
      setLabsImageFile(null);
      setLabsImagePreviewUrl(null);
    }
  }, []);

  const renderApiKeyStatus = () => {
    if (showSelectKeyButton) {
      return (
        <div className="bg-blue-100 border-l-4 border-blue-500 text-blue-700 p-4 mb-6 rounded-lg shadow-md flex items-center justify-between">
          <p className="font-medium">{apiKeyStatusMessage}</p>
          <button
            onClick={handleSelectApiKey}
            className="ml-4 bg-primary hover:bg-primary-hover text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-75 transition duration-150 ease-in-out"
            disabled={apiIsSelecting}
          >
            {apiIsSelecting ? 'Selecting...' : 'Select API Key'}
          </button>
        </div>
      );
    }
    if (!apiConfigured && !process.env.API_KEY) {
      return (
        <div className="bg-red-100 border-l-4 border-error text-red-700 p-4 mb-6 rounded-lg shadow-md">
          <p className="font-medium">{apiKeyStatusMessage}</p>
        </div>
      );
    }
    return null; // Key is configured, no message needed
  };

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-2xl p-8 transform transition-all duration-300 ease-in-out">
      <h1 className="text-4xl font-extrabold text-center text-primary mb-8 animate-fade-in">
        <span className="bg-gradient-to-r from-purple-600 to-indigo-600 text-transparent bg-clip-text">Patient Advocate AI</span>
      </h1>

      {renderApiKeyStatus()}

      <Disclaimer>
        This AI is for informational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for any health concerns.
      </Disclaimer>

      <div className="flex justify-center mb-8 bg-gray-100 rounded-lg p-2 shadow-inner">
        {['translate', 'prepare', 'summarize', 'labs', 'record'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab as TabName)}
            className={`px-6 py-3 mx-1 rounded-md text-lg font-medium transition-all duration-300 ease-in-out
              ${activeTab === tab
                ? 'bg-primary text-white shadow-lg transform scale-105'
                : 'bg-transparent text-textSecondary hover:bg-gray-200 hover:text-textPrimary'
              }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      <div className="tab-content bg-bgSecondary p-6 rounded-lg shadow-lg">
        {activeTab === 'translate' && (
          <div>
            <h2 className="text-2xl font-bold text-primary mb-4">Translate Medical Jargon</h2>
            <p className="text-textSecondary mb-4">
              Enter medical terms or phrases, and I'll translate them into simple, everyday language.
            </p>
            <textarea
              className="w-full p-3 border border-borderColor rounded-md focus:outline-none focus:ring-2 focus:ring-primary mb-3 text-textPrimary h-32 resize-y"
              placeholder="e.g., 'Patient presents with severe dyspnea and persistent dysphagia.'"
              value={translateInput}
              onChange={(e) => setTranslateInput(e.target.value)}
            ></textarea>
            <div className="flex space-x-2 mb-4">
              <button
                onClick={handleTranslateJargon}
                className="bg-primary hover:bg-primary-hover text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-75 transition duration-150 ease-in-out flex items-center justify-center"
                disabled={translateLoading}
              >
                {translateLoading && <Spinner size="w-4 h-4" color="border-t-white" className="mr-2" />}
                Translate
              </button>
              <button
                onClick={() => proofreadText({
                  textToProofread: translateInput,
                  setLoading: setTranslateProofreadingLoading,
                  setError: setTranslateProofreadingError,
                  setResult: setTranslateProofreadingResult,
                })}
                className="bg-secondary hover:bg-emerald-600 text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-secondary focus:ring-opacity-75 transition duration-150 ease-in-out flex items-center justify-center"
                disabled={translateProofreadingLoading}
              >
                {translateProofreadingLoading && <Spinner size="w-4 h-4" color="border-t-white" className="mr-2" />}
                Proofread Input
              </button>
            </div>


            {translateError && <ErrorMessage message={translateError} />}
            {translateProofreadingError && <ErrorMessage message={`Proofreading Error: ${translateProofreadingError}`} />}

            {translateProofreadingResult && (
              <ResultSection
                title="Proofread Input"
                content={renderMarkdown(translateProofreadingResult)}
                variant="info"
              />
            )}
            {translateOutput && (
              <ResultSection
                title="Translated Explanation"
                content={renderMarkdown(translateOutput)}
              />
            )}
          </div>
        )}

        {activeTab === 'prepare' && (
          <div>
            <h2 className="text-2xl font-bold text-primary mb-4">Prepare for Appointment</h2>
            <p className="text-textSecondary mb-4">
              Enter your concerns, symptoms, or questions, and I'll help you organize them and suggest
              additional questions to ask your doctor.
            </p>
            <textarea
              className="w-full p-3 border border-borderColor rounded-md focus:outline-none focus:ring-2 focus:ring-primary mb-3 text-textPrimary h-32 resize-y"
              placeholder="e.g., 'I've been feeling tired, headaches often, stomach hurts sometimes, and I'm worried about my blood pressure.'"
              value={prepareInput}
              onChange={(e) => setPrepareInput(e.target.value)}
            ></textarea>
            <div className="flex space-x-2 mb-4">
              <button
                onClick={handlePrepareAppointment}
                className="bg-primary hover:bg-primary-hover text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-75 transition duration-150 ease-in-out flex items-center justify-center"
                disabled={prepareLoading}
              >
                {prepareLoading && <Spinner size="w-4 h-4" color="border-t-white" className="mr-2" />}
                Prepare
              </button>
              <button
                onClick={() => proofreadText({
                  textToProofread: prepareInput,
                  setLoading: setPrepareProofreadingLoading,
                  setError: setPrepareProofreadingError,
                  setResult: setPrepareProofreadingResult,
                })}
                className="bg-secondary hover:bg-emerald-600 text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-secondary focus:ring-opacity-75 transition duration-150 ease-in-out flex items-center justify-center"
                disabled={prepareProofreadingLoading}
              >
                {prepareProofreadingLoading && <Spinner size="w-4 h-4" color="border-t-white" className="mr-2" />}
                Proofread Input
              </button>
            </div>
            {prepareError && <ErrorMessage message={prepareError} />}
            {prepareProofreadingError && <ErrorMessage message={`Proofreading Error: ${prepareProofreadingError}`} />}

            {prepareProofreadingResult && (
              <ResultSection
                title="Proofread Input"
                content={renderMarkdown(prepareProofreadingResult)}
                variant="info"
              />
            )}
            {prepareOutput && (
              <ResultSection
                title="Appointment Preparation"
                content={renderMarkdown(prepareOutput)}
              />
            )}
            {prepareGroundingUrls && prepareGroundingUrls.length > 0 && (
              <ResultSection
                title="References"
                variant="info"
                content={
                  <ul className="list-disc pl-5">
                    {prepareGroundingUrls.map((url, index) => (
                      <li key={index} className="mb-1">
                        <a href={url.uri} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                          {url.title || url.uri}
                        </a>
                      </li>
                    ))}
                  </ul>
                }
              />
            )}
          </div>
        )}

        {activeTab === 'summarize' && (
          <div>
            <h2 className="text-2xl font-bold text-primary mb-4">Summarize Medical Notes</h2>
            <p className="text-textSecondary mb-4">
              Upload an image of your notes or type them in, and I'll provide a concise summary,
              including any mentioned medications.
            </p>
            <div className="flex items-center space-x-4 mb-4">
              <input
                type="file"
                accept="image/*"
                onChange={handleSummarizeImageChange}
                className="block w-full text-sm text-textPrimary file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-primary hover:file:bg-violet-100"
              />
              {summarizeImagePreviewUrl && (
                <img src={summarizeImagePreviewUrl} alt="Notes Preview" className="h-24 w-24 object-cover rounded-md shadow-md" />
              )}
            </div>
            <textarea
              className="w-full p-3 border border-borderColor rounded-md focus:outline-none focus:ring-2 focus:ring-primary mb-3 text-textPrimary h-32 resize-y"
              placeholder="e.g., 'Patient seen for follow-up on hypertension. BP 140/90. Current medications: Lisinopril 10mg daily. Discussed diet and exercise.'"
              value={summarizeInput}
              onChange={(e) => setSummarizeInput(e.target.value)}
            ></textarea>
            <div className="flex space-x-2 mb-4">
              <button
                onClick={handleSummarizeNotes}
                className="bg-primary hover:bg-primary-hover text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-75 transition duration-150 ease-in-out flex items-center justify-center"
                disabled={summarizeLoading || (!summarizeInput.trim() && !summarizeImageFile)}
              >
                {summarizeLoading && <Spinner size="w-4 h-4" color="border-t-white" className="mr-2" />}
                Summarize
              </button>
              <button
                onClick={() => proofreadText({
                  textToProofread: summarizeInput,
                  setLoading: setSummarizeProofreadingLoading,
                  setError: setSummarizeProofreadingError,
                  setResult: setSummarizeProofreadingResult,
                })}
                className="bg-secondary hover:bg-emerald-600 text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-secondary focus:ring-opacity-75 transition duration-150 ease-in-out flex items-center justify-center"
                disabled={summarizeProofreadingLoading || !summarizeInput.trim()}
              >
                {summarizeProofreadingLoading && <Spinner size="w-4 h-4" color="border-t-white" className="mr-2" />}
                Proofread Input
              </button>
            </div>
            {summarizeError && <ErrorMessage message={summarizeError} />}
            {summarizeProofreadingError && <ErrorMessage message={`Proofreading Error: ${summarizeProofreadingError}`} />}

            {summarizeProofreadingResult && (
              <ResultSection
                title="Proofread Input"
                content={renderMarkdown(summarizeProofreadingResult)}
                variant="info"
              />
            )}
            {summarizeOutput && (
              <ResultSection
                title="Summary"
                content={renderMarkdown(summarizeOutput)}
              />
            )}
            {extractedMedications && (
              <ResultSection
                title="Extracted Medications"
                content={renderMarkdown(extractedMedications)}
                variant="info"
              />
            )}
          </div>
        )}

        {activeTab === 'labs' && (
          <div>
            <h2 className="text-2xl font-bold text-primary mb-4">Explain Lab Results</h2>
            <p className="text-textSecondary mb-4">
              Upload an image or type in your lab results, and I'll explain what they mean in simple terms.
            </p>
            <div className="flex items-center space-x-4 mb-4">
              <input
                type="file"
                accept="image/*"
                onChange={handleLabsImageChange}
                className="block w-full text-sm text-textPrimary file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-primary hover:file:bg-violet-100"
              />
              {labsImagePreviewUrl && (
                <img src={labsImagePreviewUrl} alt="Labs Preview" className="h-24 w-24 object-cover rounded-md shadow-md" />
              )}
            </div>
            <textarea
              className="w-full p-3 border border-borderColor rounded-md focus:outline-none focus:ring-2 focus:ring-primary mb-3 text-textPrimary h-32 resize-y"
              placeholder="e.g., 'Glucose: 120 mg/dL (High), Hemoglobin A1c: 6.8% (High), Cholesterol Total: 220 mg/dL (High)'"
              value={labsInput}
              onChange={(e) => setLabsInput(e.target.value)}
            ></textarea>
            <div className="flex space-x-2 mb-4">
              <button
                onClick={handleLabsExplanation}
                className="bg-primary hover:bg-primary-hover text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-primary focus:ring-opacity-75 transition duration-150 ease-in-out flex items-center justify-center"
                disabled={labsLoading || (!labsInput.trim() && !labsImageFile)}
              >
                {labsLoading && <Spinner size="w-4 h-4" color="border-t-white" className="mr-2" />}
                Explain
              </button>
              <button
                onClick={() => proofreadText({
                  textToProofread: labsInput,
                  setLoading: setLabsProofreadingLoading,
                  setError: setLabsProofreadingError,
                  setResult: setLabsProofreadingResult,
                })}
                className="bg-secondary hover:bg-emerald-600 text-white font-bold py-2 px-4 rounded-md focus:outline-none focus:ring-2 focus:ring-secondary focus:ring-opacity-75 transition duration-150 ease-in-out flex items-center justify-center"
                disabled={labsProofreadingLoading || !labsInput.trim()}
              >
                {labsProofreadingLoading && <Spinner size="w-4 h-4" color="border-t-white" className="mr-2" />}
                Proofread Input
              </button>
            </div>
            {labsError && <ErrorMessage message={labsError} />}
            {labsProofreadingError && <ErrorMessage message={`Proofreading Error: ${labsProofreadingError}`} />}

            {labsProofreadingResult && (
              <ResultSection
                title="Proofread Input"
                content={renderMarkdown(labsProofreadingResult)}
                variant="info"
              />
            )}
            {labsOutput && (
              <ResultSection
                title="Lab Results Explanation"
                content={renderMarkdown(labsOutput)}
              />
            )}
            {labsGroundingUrls && labsGroundingUrls.length > 0 && (
              <ResultSection
                title="References"
                variant="info"
                content={
                  <ul className="list-disc pl-5">
                    {labsGroundingUrls.map((url, index) => (
                      <li key={index} className="mb-1">
                        <a href={url.uri} target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">
                          {url.title || url.uri}
                        </a>
                      </li>
                    ))}
                  </ul>
                }
              />
            )}
          </div>
        )}

        {activeTab === 'record' && (
          <div className="relative">
            <h2 className="text-2xl font-bold text-primary mb-4">Record Session</h2>
            <p className="text-textSecondary mb-4">
              Have a real-time conversation with your AI medical advocate. Speak your concerns or questions, and the AI will respond.
            </p>

            <div className="flex justify-center mb-6">
              {!isRecording ? (
                <button
                  onClick={handleStartRecording}
                  className="bg-green-500 hover:bg-green-600 text-white font-bold py-3 px-6 rounded-full text-lg shadow-md focus:outline-none focus:ring-4 focus:ring-green-300 transition duration-300 ease-in-out transform hover:scale-105 flex items-center"
                  disabled={liveLoading}
                >
                  {liveLoading && <Spinner size="w-5 h-5" color="border-t-white" className="mr-3" />}
                  <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a7 7 0 01-7-7m7 7a7 7 0 007-7m0 0H8m4-4h4m-4-4H8"></path></svg>
                  Start Recording
                </button>
              ) : (
                <button
                  onClick={handleStopRecording}
                  className="bg-red-500 hover:bg-red-600 text-white font-bold py-3 px-6 rounded-full text-lg shadow-md focus:outline-none focus:ring-4 focus:ring-red-300 transition duration-300 ease-in-out transform hover:scale-105 flex items-center"
                  disabled={!isRecording}
                >
                  <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"></path></svg>
                  Stop Recording
                </button>
              )}
            </div>

            {liveLoading && isRecording && (
              <div className="flex items-center justify-center mb-4 text-primary font-medium">
                <Spinner size="w-6 h-6" className="mr-3" />
                Listening...
              </div>
            )}

            {liveError && <ErrorMessage message={liveError} />}

            <div className="mt-6 border border-borderColor rounded-lg p-4 bg-white shadow-inner max-h-96 overflow-y-auto">
              <h3 className="text-xl font-semibold mb-3 text-primary">Conversation Log</h3>
              {transcriptionHistory.length === 0 && !currentLiveInputTranscription && !currentLiveOutputTranscription ? (
                <p className="text-textSecondary text-center italic">Start recording to begin your conversation.</p>
              ) : (
                transcriptionHistory.map((entry, index) => (
                  <div key={index} className={`mb-2 p-2 rounded-lg ${entry.speaker === 'User' ? 'bg-blue-50 text-blue-800 self-end text-right ml-auto' : 'bg-green-50 text-green-800 self-start mr-auto'}`}>
                    <strong className="font-semibold">{entry.speaker}:</strong> {entry.text}
                  </div>
                ))
              )}
              {currentLiveInputTranscription && (
                <div className="mb-2 p-2 rounded-lg bg-blue-50 text-blue-800 self-end text-right ml-auto animate-pulse">
                  <strong className="font-semibold">User (Live):</strong> {currentLiveInputTranscription}
                </div>
              )}
              {currentLiveOutputTranscription && (
                <div className="mb-2 p-2 rounded-lg bg-green-50 text-green-800 self-start mr-auto animate-pulse">
                  <strong className="font-semibold">Model (Live):</strong> {currentLiveOutputTranscription}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default App;
