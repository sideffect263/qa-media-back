// src/services/qaService.mjs
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { promises as fs } from 'fs';
import { spawn } from 'child_process';
import EventEmitter from 'events';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class QAService extends EventEmitter {
    constructor() {
        super();
    }

    async answerQuestion(mediaId, question) {
        try {
            this.emit('progress', { 
                stage: 'preparation', 
                progress: 0,
                message: 'Preparing QA system'
            });
            
            // Load transcript
            this.emit('progress', { 
                stage: 'loading_transcript', 
                progress: 25,
                message: 'Loading transcript'
            });

            const transcriptPath = join(__dirname, '../../uploads', `${mediaId}.transcript`);
            
            try {
                await fs.access(transcriptPath);
            } catch (error) {
                throw new Error(`Transcript not found. Please ensure the audio has been processed. Error: ${error.message}`);
            }

            this.emit('progress', { 
                stage: 'processing', 
                progress: 50,
                message: 'Processing question'
            });

            const transcript = await fs.readFile(transcriptPath, 'utf-8');
            if (!transcript.trim()) {
                throw new Error('Transcript is empty');
            }

            // Call Python script
            const result = await new Promise((resolve, reject) => {
                const pythonProcess = spawn('python', [
                    join(__dirname, 'qa.py'),
                    question,
                    transcriptPath
                ]);

                let outputData = '';
                let errorData = '';

                pythonProcess.stdout.on('data', (data) => {
                    outputData += data.toString();
                    console.log('Python QA output:', data.toString());
                });

                pythonProcess.stderr.on('data', (data) => {
                    errorData += data.toString();
                    console.error('Python QA error:', errorData);
                });

                pythonProcess.on('close', (code) => {
                    if (code !== 0) {
                        reject(new Error(`QA failed: ${errorData}`));
                        return;
                    }

                    try {
                        const result = JSON.parse(outputData);
                        if (result.error) {
                            reject(new Error(result.error));
                            return;
                        }
                        resolve(result);
                    } catch (error) {
                        reject(new Error(`Failed to parse QA result: ${error.message}`));
                    }
                });
            });

            this.emit('progress', { 
                stage: 'complete', 
                progress: 100,
                message: 'Answer generated'
            });

            return {
                answer: result.answer,
                confidence: result.confidence,
                context: transcript.substring(
                    Math.max(0, result.start - 50),
                    Math.min(transcript.length, result.end + 50)
                )
            };

        } catch (error) {
            console.error('QA Processing error:', error);
            this.emit('progress', {
                stage: 'error',
                progress: 0,
                message: `Question answering failed: ${error.message}`
            });
            throw error;
        }
    }

    async getTranscript(mediaId) {
        try {
            const transcriptPath = join(__dirname, '../../uploads', `${mediaId}.transcript`);
            return await fs.readFile(transcriptPath, 'utf-8');
        } catch (error) {
            throw new Error(`Failed to read transcript: ${error.message}`);
        }
    }
}

// Create a single instance and export it
const qaService = new QAService();
export { qaService };

// Also export the class if needed
export default QAService;