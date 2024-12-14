// src/services/transcriptionService.mjs
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { promises as fs } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const transcribeAudio = async (audioPath) => {
  return new Promise((resolve, reject) => {
    // Create transcript file path
    const transcriptPath = audioPath + '.transcript';
    
    // Spawn Python process
    const pythonProcess = spawn('python', [
      path.join(__dirname, 'transcribe.py'),
      audioPath
    ]);

    let outputData = '';
    let errorData = '';

    pythonProcess.stdout.on('data', (data) => {
      outputData += data.toString();
      console.log('Transcription output:', outputData);
    });

    pythonProcess.stderr.on('data', (data) => {
      errorData += data.toString();
      console.error('Transcription error:', errorData);
    });

    pythonProcess.on('close', async (code) => {
      console.log('Python process exit code:', code);

      if (code !== 0) {
        reject(new Error(`Transcription failed: ${errorData}`));
        return;
      }

      try {
        const result = JSON.parse(outputData);
        if (result.error) {
          reject(new Error(result.error));
          return;
        }

        // Save transcript to file
        await fs.writeFile(transcriptPath, result.text, 'utf-8');
        console.log('Transcript saved to:', transcriptPath);

        resolve({
          ...result,
          transcriptPath
        });
      } catch (error) {
        reject(new Error(`Failed to process transcription result: ${error.message}`));
      }
    });
  });
};

export default {
  transcribeAudio
};