import { GoogleGenAI, GenerateContentResponse, Part } from "@google/genai";

const GEMINI_MODEL = 'gemini-2.5-flash'; // Default model for text-only

interface CallGeminiParams {
  prompt: string;
  systemInstruction?: string;
  useSearchGrounding?: boolean;
  imageFile?: File; // Optional image file for multimodal input
}

export interface GeminiResponseData {
  text: string;
  groundingUrls?: { uri: string; title?: string }[];
}

/**
 * Converts a File object to a Base64 string and returns its data and MIME type.
 * @param file The File object to convert.
 * @returns A Promise resolving to an object with the Base64 data and MIME type.
 */
const fileToBase64 = (file: File): Promise<{ data: string; mimeType: string }> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const base64String = reader.result as string;
      const data = base64String.split(',')[1]; // Extract base64 data after "data:image/jpeg;base64,"
      resolve({ data, mimeType: file.type });
    };
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(file);
  });
};

export async function callGemini(params: CallGeminiParams): Promise<GeminiResponseData> {
  const { prompt, systemInstruction, useSearchGrounding = false, imageFile } = params;
  const apiKey = process.env.API_KEY;

  if (!apiKey) {
    throw new Error(
      `Gemini API key is not configured. Please set the API_KEY environment variable. 
      For local development, you might need to configure it in your build tool (e.g., VITE_API_KEY for Vite apps).`
    );
  }

  const ai = new GoogleGenAI({ apiKey });

  let modelToUse: string;

  if (useSearchGrounding) {
    // If search grounding is requested, use a model known to support it,
    // and which is also suitable for potentially complex queries and multimodal input.
    // gemini-2.5-pro is generally more capable for tasks requiring grounding and can handle images.
    modelToUse = 'gemini-2.5-pro';
  } else if (imageFile) {
    // If there's an image but no search grounding, use the dedicated multimodal image model.
    modelToUse = 'gemini-2.5-flash-image';
  } else {
    // Default for text-only, no search grounding.
    modelToUse = GEMINI_MODEL; // which is 'gemini-2.5-flash'
  }

  try {
    const config: any = {
      temperature: 0.7,
      maxOutputTokens: 2048,
      ...(systemInstruction && { systemInstruction }),
    };

    if (useSearchGrounding) {
      config.tools = [{ googleSearch: {} }];
      // As per guidelines, ensure responseMimeType and responseSchema are NOT set when googleSearch is used.
    }

    // Build content parts
    const parts: Part[] = [];
    if (imageFile) {
      const imageData = await fileToBase64(imageFile);
      parts.push({
        inlineData: {
          data: imageData.data,
          mimeType: imageData.mimeType,
        },
      });
    }
    parts.push({ text: prompt });

    const response: GenerateContentResponse = await ai.models.generateContent({
      model: modelToUse,
      contents: [{ parts: parts }],
      config: config
    });

    if (!response || !response.text) {
      // More detailed check if text is missing
      if (response && response.promptFeedback?.blockReason) {
        throw new Error(`Gemini API response blocked due to prompt feedback: ${response.promptFeedback.blockReason}`);
      }
      if (response && response.candidates && response.candidates.length > 0) {
          const candidate = response.candidates[0];
          if (candidate.finishReason === 'SAFETY') {
              throw new Error("Gemini API response blocked due to safety reasons in candidate output.");
          }
          if (candidate.finishReason === 'STOP' && (!candidate.content?.parts || candidate.content.parts.length === 0)) {
               throw new Error("Gemini API generated an empty text response (no content parts).");
          }
      }
      throw new Error("No text content found in Gemini API response or empty response with no specific block reason.");
    }

    const result: GeminiResponseData = {
      text: response.text,
    };

    if (useSearchGrounding && response.candidates?.[0]?.groundingMetadata?.groundingChunks) {
      result.groundingUrls = response.candidates[0].groundingMetadata.groundingChunks
        .filter(chunk => chunk.web?.uri)
        .map(chunk => ({
          uri: chunk.web!.uri!,
          title: chunk.web!.title || 'External Link',
        }));
    }

    return result;
  } catch (error: unknown) {
    console.error('Gemini API Error:', error);
    if (error instanceof Error) {
      throw new Error(`Gemini API request failed: ${error.message}`);
    }
    throw new Error("An unknown error occurred during Gemini API call.");
  }
}


/**
 * Proofreads the given text for grammar, spelling, and punctuation errors.
 * Returns the corrected text.
 * @param text The text to proofread.
 * @returns A Promise resolving to the corrected text.
 */
export async function callGeminiProofread(text: string): Promise<string> {
  const apiKey = process.env.API_KEY;

  if (!apiKey) {
    throw new Error("Gemini API key is not configured for proofreading.");
  }

  const ai = new GoogleGenAI({ apiKey });

  try {
    const prompt = `Proofread the following text for grammar, spelling, and punctuation errors. Return only the corrected text. If no corrections are needed, return the original text exactly as provided.

Text to proofread:
"${text}"`;

    const response: GenerateContentResponse = await ai.models.generateContent({
      model: 'gemini-2.5-flash', // Using flash for speed
      contents: [{ parts: [{ text: prompt }] }],
      config: {
        temperature: 0.2, // Lower temperature for more deterministic corrections
        maxOutputTokens: 2048,
      }
    });

    if (!response || !response.text) {
      // If the model returns nothing, consider the original text as correct or an empty string as a no-op
      // For proofreading, an empty response effectively means no changes, or an error.
      // We'll return original text as a fallback if no specific error occurred within the API call itself.
      if (response && response.promptFeedback?.blockReason) {
        throw new Error(`Proofread API response blocked due to prompt feedback: ${response.promptFeedback.blockReason}`);
      }
      if (response && response.candidates && response.candidates.length > 0) {
          const candidate = response.candidates[0];
          if (candidate.finishReason === 'SAFETY') {
              throw new Error("Proofread API response blocked due to safety reasons in candidate output.");
          }
          if (candidate.finishReason === 'STOP' && (!candidate.content?.parts || candidate.content.parts.length === 0)) {
               console.warn("Proofread API generated an empty text response (no content parts), returning original text.");
               return text; // Return original text if model produced no output
          }
      }
      console.warn("Proofread API returned no text content, returning original text.");
      return text;
    }

    return response.text.trim();
  } catch (error: unknown) {
    console.error('Gemini Proofread API Error:', error);
    if (error instanceof Error) {
      throw new Error(`Gemini Proofread API request failed: ${error.message}`);
    }
    throw new Error("An unknown error occurred during Gemini Proofread API call.");
  }
}